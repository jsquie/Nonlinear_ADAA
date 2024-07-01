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

const MAX_BLOCK_SIZE: usize = 32;
const MAX_OS_FACTOR_SCALE: usize = 16;
const PEAK_METER_DECAY_MS: f64 = 150.0;
const PEAK_DECAY_FACTOR: f64 = 0.05;

pub struct NonlinearAdaa {
    params: Arc<NonlinearAdaaParams>,
    non_linear_processors: [NonlinearProcessor; 2],
    proc_state: ProcessorState,
    oversamplers: [Oversample; 2],
    over_sample_process_buf: [[f32; MAX_BLOCK_SIZE * MAX_OS_FACTOR_SCALE]; 2],
    pre_filters: [IIRBiquadFilter; 2],
    peak_meter_decay_weight: f32,
    input_meters: [Arc<AtomicF32>; 2],
    output_meters: [Arc<AtomicF32>; 2],
    mix_scratch_buffer: [[f32; MAX_BLOCK_SIZE]; 2],
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
            over_sample_process_buf: [[0.0; MAX_OS_FACTOR_SCALE * MAX_BLOCK_SIZE]; 2],
            pre_filters: [IIRBiquadFilter::default(), IIRBiquadFilter::default()],
            peak_meter_decay_weight: 1.0,
            input_meters: [
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            ],
            output_meters: [
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            ],
            mix_scratch_buffer: [[0.0_f32; MAX_BLOCK_SIZE]; 2],
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
                22049.0,
                FloatRange::Skewed {
                    min: 50.,
                    max: 22049.0,
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

#[inline]
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
            .for_each(|os| os.copy_from_slice(&[0.0; MAX_BLOCK_SIZE * MAX_OS_FACTOR_SCALE]));

        self.mix_scratch_buffer
            .iter_mut()
            .for_each(|m_buff| m_buff.copy_from_slice(&[0.0; MAX_BLOCK_SIZE]));

        self.pre_filters.iter_mut().for_each(|x| x.reset());
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if !self.params.bypass.value() {
            let current_os_factor = self.params.os_level.value();

            // check os factor paramater -- if different reset oversample stages and set dry delay
            // amount
            self.oversamplers
                .iter_mut()
                .zip(self.dry_delay.iter_mut())
                .for_each(|(os, d)| {
                    if current_os_factor != os.get_oversample_factor() {
                        os.set_oversample_factor(current_os_factor);
                        d.set_delay_len(os.get_latency_samples());
                    }
                });

            // determine current nonlinear state from user params
            let p_state = State(
                self.params.nl_proc_type.value(),
                self.params.nl_proc_order.value(),
            );

            // change the nonlinear procressors if params are different
            self.non_linear_processors.iter_mut().for_each(|nl| {
                nl.compare_and_change_state(p_state);
            });

            // report latency of oversample FIR filters to DAW
            context.set_latency_samples(self.oversamplers[0].get_latency_samples() as u32);

            // to determine how many samples to process, given current oversample factor
            let samples_to_take = 2_u32.pow(self.oversamplers[0].get_oversample_factor() as u32)
                as usize
                * buffer.samples();

            let num_samples = buffer.samples();

            let mut left_in_amplitude = 0.0;
            let mut right_in_amplitude = 0.0;
            let mut left_out_amplitude = 0.0;
            let mut right_out_amplitude = 0.0;

            for (_, block) in buffer.iter_blocks(MAX_BLOCK_SIZE) {
                let mut block_channels = block.into_iter();
                let left = block_channels.next().unwrap();
                let right = block_channels.next().unwrap();

                self.mix_scratch_buffer[0].copy_from_slice(left);
                self.mix_scratch_buffer[1].copy_from_slice(right);

                // prefilter processing
                for (l, r) in left.iter_mut().zip(right.iter_mut()) {
                    let param_pre_filter_cutoff: &Smoother<f32> =
                        &self.params.pre_filter_cutoff.smoothed;

                    // recalculate every coefficient while smoothing
                    if param_pre_filter_cutoff.is_smoothing() {
                        let next_smoothed = param_pre_filter_cutoff.next();
                        self.pre_filters.iter_mut().for_each(|f| {
                            f.set_cutoff(next_smoothed);
                        })
                    }

                    self.pre_filters[0].process_sample(l);
                    self.pre_filters[1].process_sample(r);
                }

                // delay the dry signal by the latency amount introduced in oversampling FIR
                // filtering
                self.dry_delay
                    .iter_mut()
                    .zip(self.mix_scratch_buffer.iter_mut())
                    .for_each(|(d, m)| d.delay(m));

                let mut left_oversample_buff = self.over_sample_process_buf[0];
                let mut right_oversample_buff = self.over_sample_process_buf[1];

                self.oversamplers[0].process_up(left, &mut left_oversample_buff);
                self.oversamplers[1].process_up(right, &mut right_oversample_buff);
                // nonlinear process oversampled signal
                left_oversample_buff
                    .iter_mut()
                    .take(samples_to_take)
                    .zip(right_oversample_buff.iter_mut())
                    .for_each(|(os_l, os_r)| {
                        let gain = self.params.gain.smoothed.next();
                        let output = self.params.output.smoothed.next();

                        *os_l *= gain;
                        *os_r *= gain;

                        left_in_amplitude += *os_l;
                        right_in_amplitude += *os_r;

                        *os_l = self.non_linear_processors[0].process(*os_l);
                        *os_r = self.non_linear_processors[1].process(*os_r);

                        *os_l *= output;
                        *os_r *= output;
                    });

                // down sample processed signal and store in block channel
                self.oversamplers[0].process_down(&mut left_oversample_buff, left);
                self.oversamplers[1].process_down(&mut right_oversample_buff, right);

                for (l_wet, (l_dry, (r_wet, r_dry))) in left.iter_mut().zip(
                    self.mix_scratch_buffer[0]
                        .iter()
                        .zip(right.iter_mut().zip(self.mix_scratch_buffer[1].iter())),
                ) {
                    let wet_amt = self.params.dry_wet.smoothed.next();
                    let dry_amt = 1.0 - wet_amt;

                    *l_wet = (wet_amt * *l_wet) + (dry_amt * l_dry);
                    *r_wet = (wet_amt * *r_wet) + (dry_amt * r_dry);

                    left_out_amplitude += *l_wet;
                    right_out_amplitude += *r_wet;
                }
            }

            // display meter levels only if GUI is open
            if self.params.editor_state.is_open() {
                left_in_amplitude = (left_in_amplitude / samples_to_take as f32).abs();
                right_in_amplitude = (right_in_amplitude / samples_to_take as f32).abs();
                left_out_amplitude = (left_out_amplitude / num_samples as f32).abs();
                right_out_amplitude = (right_out_amplitude / num_samples as f32).abs();

                let before_left_input_meter = self.input_meters[0].load(Ordering::Relaxed);
                let before_right_input_meter = self.input_meters[1].load(Ordering::Relaxed);
                let before_left_out_meter = self.output_meters[0].load(Ordering::Relaxed);
                let before_right_out_meter = self.output_meters[1].load(Ordering::Relaxed);

                let new_left_input_meter = if left_in_amplitude > before_left_input_meter {
                    left_in_amplitude.clamp(0., 10.0)
                } else {
                    before_left_input_meter * self.peak_meter_decay_weight
                        + left_in_amplitude * (1.0 - self.peak_meter_decay_weight)
                };

                let new_right_input_meter = if right_in_amplitude > before_right_input_meter {
                    right_in_amplitude.clamp(0., 10.0)
                } else {
                    before_right_input_meter * self.peak_meter_decay_weight
                        + right_in_amplitude * (1.0 - self.peak_meter_decay_weight)
                };

                let new_left_out_meter = if left_out_amplitude > before_left_out_meter {
                    left_out_amplitude.clamp(0., 10.0)
                } else {
                    before_left_out_meter * (self.peak_meter_decay_weight)
                        + left_out_amplitude * (1.0 - self.peak_meter_decay_weight)
                };

                let new_right_out_meter = if right_out_amplitude > before_right_out_meter {
                    right_out_amplitude.clamp(0., 10.0)
                } else {
                    before_right_out_meter * (self.peak_meter_decay_weight)
                        + right_out_amplitude * (1.0 - self.peak_meter_decay_weight)
                };

                self.input_meters[0].store(new_left_input_meter, Ordering::Relaxed);
                self.input_meters[1].store(new_right_input_meter, Ordering::Relaxed);
                self.output_meters[0].store(new_left_out_meter, Ordering::Relaxed);
                self.output_meters[1].store(new_right_out_meter, Ordering::Relaxed);
            } else {
                self.input_meters[0].store(0.0, Ordering::Relaxed);
                self.input_meters[1].store(0.0, Ordering::Relaxed);
                self.output_meters[0].store(0.0, Ordering::Relaxed);
                self.output_meters[1].store(0.0, Ordering::Relaxed);
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
