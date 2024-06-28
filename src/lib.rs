use itertools::izip;
use jdsp::{
    AntiderivativeOrder, CircularDelayBuffer, NonlinearProcessor, ProcessorState,
    ProcessorState::State, ProcessorStyle, MAX_LATENCY_AMT,
};
use jdsp::{FilterOrder, IIRBiquadFilter};
use jdsp::{Oversample, OversampleFactor};
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use std::sync::atomic::Ordering;
use std::sync::Arc;

mod editor;
mod knob;

const MAX_BLOCK_SIZE: usize = 32;
const MAX_OS_FACTOR_SCALE: usize = 16;
const PEAK_METER_DECAY_MS: f64 = 150.0;
const PEAK_DECAY_FACTOR: f64 = 0.05;

pub struct NonlinearAdaa {
    params: Arc<NonlinearAdaaParams>,
    non_linear_processors: [NonlinearProcessor; 2],
    proc_state: ProcessorState,
    oversamplers: [Oversample; 2],
    over_sample_process_buf: [f32; MAX_BLOCK_SIZE * MAX_OS_FACTOR_SCALE],
    pre_filters: [IIRBiquadFilter; 2],
    peak_meter_decay_weight: f32,
    peak_meter_decay_scale_pos: usize,
    input_meters: [Arc<AtomicF32>; 2],
    output_meters: [Arc<AtomicF32>; 2],
    mix_scratch_buffer: [f32; MAX_BLOCK_SIZE],
    dry_delay: [CircularDelayBuffer; 2],
}

#[derive(Params, Debug)]
pub struct NonlinearAdaaParams {
    #[id = "gain"]
    pub gain: FloatParam,
    #[id = "output"]
    pub output: FloatParam,
    #[id = "nl proc"]
    pub nl_proc_type: EnumParam<ProcessorStyle>,
    #[id = "ad level"]
    pub nl_proc_order: EnumParam<AntiderivativeOrder>,
    #[id = "os level"]
    pub os_level: EnumParam<OversampleFactor>,
    #[id = "pre filter cutoff"]
    pub pre_filter_cutoff: FloatParam,
    #[id = "dry wet"]
    pub dry_wet: FloatParam,
    #[id = "plugin bypass"]
    pub bypass: BoolParam,
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,
}

impl Default for NonlinearAdaa {
    fn default() -> Self {
        Self {
            params: Arc::new(NonlinearAdaaParams::new()),
            non_linear_processors: [NonlinearProcessor::new(), NonlinearProcessor::new()],
            proc_state: State(ProcessorStyle::HardClip, AntiderivativeOrder::FirstOrder),
            oversamplers: [
                Oversample::new(OversampleFactor::TwoTimes, MAX_BLOCK_SIZE),
                Oversample::new(OversampleFactor::TwoTimes, MAX_BLOCK_SIZE),
            ],
            over_sample_process_buf: [0.0; MAX_OS_FACTOR_SCALE * MAX_BLOCK_SIZE],
            pre_filters: [IIRBiquadFilter::default(), IIRBiquadFilter::default()],
            peak_meter_decay_weight: 1.0,
            peak_meter_decay_scale_pos: 0,
            input_meters: [
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            ],
            output_meters: [
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            ],
            mix_scratch_buffer: [0.0_f32; MAX_BLOCK_SIZE],
            dry_delay: [
                CircularDelayBuffer::new(MAX_LATENCY_AMT),
                CircularDelayBuffer::new(MAX_LATENCY_AMT),
            ],
        }
    }
}

impl NonlinearAdaaParams {
    fn new() -> Self {
        let oversampling_times = Arc::new(AtomicF32::new(oversampling_factor_to_times(
            OversampleFactor::TwoTimes,
        )));

        Self {
            editor_state: editor::default_state(),

            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(0.0),
                    max: util::db_to_gain(60.0),
                    factor: FloatRange::gain_skew_factor(0.0, 60.0),
                },
            )
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversampling_times.clone(),
                &SmoothingStyle::Logarithmic(1000.0),
            ))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),

            output: FloatParam::new(
                "Output Gain",
                util::db_to_gain(-1.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-60.0),
                    max: util::db_to_gain(0.0),
                    factor: FloatRange::gain_skew_factor(-60.0, 0.0),
                },
            )
            .with_unit(" dB")
            // The value does not go down to 0 so we can do logarithmic here
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversampling_times.clone(),
                &SmoothingStyle::Logarithmic(5.0),
            )),

            nl_proc_type: EnumParam::new("Nonlinear Process", ProcessorStyle::HardClip),

            nl_proc_order: EnumParam::new("Antiderivative Order", AntiderivativeOrder::FirstOrder),

            dry_wet: FloatParam::new("Mix Amount", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(50.0))
                .with_value_to_string(formatters::v2s_f32_percentage(1))
                .with_string_to_value(formatters::s2v_f32_percentage()),

            os_level: EnumParam::new("Oversample Factor", OversampleFactor::TwoTimes)
                .with_callback(Arc::new(move |new_factor| {
                    oversampling_times.store(
                        oversampling_factor_to_times(new_factor) as f32,
                        Ordering::Relaxed,
                    );
                })),

            pre_filter_cutoff: FloatParam::new(
                "Prefilter Cutoff Frequency",
                20000.0,
                FloatRange::Skewed {
                    min: 100.,
                    max: 22000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(10.0))
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),

            bypass: BoolParam::new("Plugin Bypass", false),
        }
    }
}

fn oversampling_factor_to_times(factor: OversampleFactor) -> f32 {
    match factor {
        OversampleFactor::TwoTimes => 2.0,
        OversampleFactor::FourTimes => 4.0,
        OversampleFactor::EightTimes => 8.0,
        OversampleFactor::SixteenTimes => 16.0,
    }
}

impl Plugin for NonlinearAdaa {
    const NAME: &'static str = "Nonlinear Adaa";
    const VENDOR: &'static str = "James Squires";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "squires.jr@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.input_meters.clone(),
            self.output_meters.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let _num_channels = _audio_io_layout
            .main_output_channels
            .expect("Plugin was initialized without any outputs")
            .get() as usize;

        self.pre_filters.iter_mut().for_each(|filter| {
            filter.init(
                &buffer_config.sample_rate,
                &self.params.pre_filter_cutoff.value(),
                FilterOrder::First,
            )
        });

        let new_state = State(
            self.params.nl_proc_type.value(),
            self.params.nl_proc_order.value(),
        );

        self.proc_state = new_state;

        self.non_linear_processors
            .iter_mut()
            .for_each(|x| *x = NonlinearProcessor::new());

        self.dry_delay
            .iter_mut()
            .zip(self.oversamplers.iter())
            .for_each(|(delay, stage)| {
                let latency = stage.get_latency_samples();
                context.set_latency_samples(latency as u32);
                delay.set_delay_len(latency)
            });

        self.peak_meter_decay_weight = PEAK_DECAY_FACTOR
            .powf((buffer_config.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;

        true
    }

    fn reset(&mut self) {
        let new_state = State(
            self.params.nl_proc_type.value(),
            self.params.nl_proc_order.value(),
        );

        self.proc_state = new_state;

        self.non_linear_processors
            .iter_mut()
            .for_each(|x| *x = NonlinearProcessor::new());

        self.oversamplers.iter_mut().for_each(|x| x.reset());
        self.over_sample_process_buf
            .iter_mut()
            .for_each(|x| *x = 0.0);
        self.pre_filters.iter_mut().for_each(|x| x.reset());
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for (_, block) in buffer.iter_blocks(MAX_BLOCK_SIZE as usize) {
            for (block_channel, oversampler, filter, nl_processor, in_meter, out_meter, delay) in izip!(
                block,
                &mut self.oversamplers,
                &mut self.pre_filters,
                &mut self.non_linear_processors,
                &mut self.input_meters,
                &mut self.output_meters,
                &mut self.dry_delay,
            ) {
                if !self.params.bypass.value() {
                    if self.params.os_level.value() != oversampler.get_oversample_factor() {
                        oversampler.set_oversample_factor(self.params.os_level.value());
                        delay.set_delay_len(oversampler.get_latency_samples());
                    };

                    let total_latency: u32 = oversampler.get_latency_samples() as u32;

                    let p_state = State(
                        self.params.nl_proc_type.value(),
                        self.params.nl_proc_order.value(),
                    );

                    // if let State(_, AntiderivativeOrder::SecondOrder) = p_state {
                    // total_latency += 1;
                    // }

                    context.set_latency_samples(total_latency);

                    self.proc_state = p_state;
                    nl_processor.compare_and_change_state(p_state);

                    // pre filter and add dry to mix_scratch_buffer
                    block_channel
                        .iter_mut()
                        .zip(self.mix_scratch_buffer.iter_mut())
                        .for_each(|(s, d)| {
                            let param_pre_filter_cutoff: &Smoother<f32> =
                                &self.params.pre_filter_cutoff.smoothed;
                            if param_pre_filter_cutoff.is_smoothing() {
                                filter.set_cutoff(param_pre_filter_cutoff.next());
                            };

                            *d = *s;
                            filter.process_sample(s);
                        });

                    // delay dry signal by total latency
                    // might have to adjust how the delay changes sizes
                    // i.e. might have to have it so that the start/finish is dynamic with the
                    // size of the latency
                    delay.delay(&mut self.mix_scratch_buffer);

                    // upsample -- store upsampled signal contents in over_sample_process_buf
                    oversampler.process_up(block_channel, &mut self.over_sample_process_buf);

                    // to determine how many samples to process, given current oversample factor
                    let samples_to_take = 2_u32.pow(oversampler.get_oversample_factor() as u32)
                        as usize
                        * MAX_BLOCK_SIZE;

                    let mut in_amplitude = 0.0;
                    let mut out_amplitude = 0.0;

                    // nonlinear process oversampled signal
                    self.over_sample_process_buf
                        .iter_mut()
                        .take(samples_to_take)
                        .for_each(|sample| {
                            let gain = self.params.gain.smoothed.next();
                            let output = self.params.output.smoothed.next();

                            *sample *= gain;
                            in_amplitude += *sample;
                            *sample = nl_processor.process(*sample);
                            *sample *= output;
                        });

                    // down sample processed signal and store in block channel
                    oversampler.process_down(&mut self.over_sample_process_buf, block_channel);

                    // mix dry with wet
                    block_channel
                        .iter_mut()
                        .zip(self.mix_scratch_buffer.iter())
                        .for_each(|(w, d)| {
                            let wet_amt = self.params.dry_wet.smoothed.next();

                            *w = (wet_amt * *w) + ((1.0 - wet_amt) * d);

                            out_amplitude += *w;
                        });

                    // display meter levels only if GUI is open
                    if self.params.editor_state.is_open() {
                        in_amplitude = (in_amplitude / samples_to_take as f32).abs();
                        out_amplitude = (out_amplitude / block_channel.len() as f32).abs();

                        let current_input_meter = in_meter.load(Ordering::Relaxed);
                        let current_out_meter = out_meter.load(Ordering::Relaxed);

                        let new_input_meter = if in_amplitude > current_input_meter {
                            self.peak_meter_decay_scale_pos = 0;
                            in_amplitude.clamp(0., 1.5)
                        } else {
                            current_input_meter * self.peak_meter_decay_weight
                                + in_amplitude * (1.0 - self.peak_meter_decay_weight)
                        };

                        let new_output_meter = if out_amplitude > current_out_meter {
                            out_amplitude.clamp(0., 1.5)
                        } else {
                            current_out_meter * (self.peak_meter_decay_weight)
                                + out_amplitude * (1.0 - self.peak_meter_decay_weight)
                        };

                        in_meter.store(new_input_meter, Ordering::Relaxed);
                        out_meter.store(new_output_meter, Ordering::Relaxed);
                    }
                }
            }
        }
        ProcessStatus::Normal
    }
}

impl ClapPlugin for NonlinearAdaa {
    const CLAP_ID: &'static str = "com.your-domain.Nonlinear-ADAA";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Nonlinear processor");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for NonlinearAdaa {
    const VST3_CLASS_ID: [u8; 16] = *b"Exactly16Chars!!";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Custom("Harmonics")];
}

nih_export_clap!(NonlinearAdaa);
nih_export_vst3!(NonlinearAdaa);
