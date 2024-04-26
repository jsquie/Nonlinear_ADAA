use adaa::NextAdaa;
use iir_biquad_filter::{FilterOrder, IIRBiquadFilter};
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use oversampler::{Oversample, OversampleFactor};
use std::sync::Arc;

mod adaa;
mod editor;

const MAX_BLOCK_SIZE: usize = 32;
const MAX_OS_FACTOR_SCALE: usize = 16;

#[derive(Enum, Debug, PartialEq)]
pub enum AntiderivativeOrder {
    #[id = "first order ad"]
    #[name = "First Order"]
    First,

    #[id = "second order ad"]
    #[name = "Second Order"]
    Second,
}

pub struct NonlinearAdaa {
    params: Arc<NonlinearAdaaParams>,
    first_order_nlprocs: Vec<adaa::ADAAFirst>,
    second_order_nlprocs: Vec<adaa::ADAASecond>,
    proc_style: adaa::NLProc,
    proc_order: AntiderivativeOrder,
    oversamplers: Vec<Oversample>,
    over_sample_process_buf: [f32; MAX_BLOCK_SIZE * MAX_OS_FACTOR_SCALE],
    pre_filters: [IIRBiquadFilter; 2],
}

#[derive(Params)]
pub struct NonlinearAdaaParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    #[id = "gain"]
    pub gain: FloatParam,
    #[id = "output"]
    pub output: FloatParam,
    #[id = "nl proc"]
    pub nl_proc_type: EnumParam<adaa::NLProc>,
    #[id = "ad level"]
    pub nl_proc_order: EnumParam<AntiderivativeOrder>,
    #[id = "os level"]
    pub os_level: EnumParam<OversampleFactor>,
    #[id = "pre filter cutoff"]
    pub pre_filter_cutoff: FloatParam,
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,
}

impl Default for NonlinearAdaa {
    fn default() -> Self {
        Self {
            params: Arc::new(NonlinearAdaaParams::default()),
            first_order_nlprocs: Vec::with_capacity(2),
            second_order_nlprocs: Vec::with_capacity(2),
            proc_style: adaa::NLProc::HardClip,
            proc_order: AntiderivativeOrder::First,
            oversamplers: Vec::with_capacity(2),
            over_sample_process_buf: [0.0; MAX_OS_FACTOR_SCALE * MAX_BLOCK_SIZE],
            pre_filters: [IIRBiquadFilter::new(), IIRBiquadFilter::new()],
        }
    }
}

impl Default for NonlinearAdaaParams {
    fn default() -> Self {
        Self {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            editor_state: editor::default_state(),
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),

            output: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-80.0),
                    max: util::db_to_gain(0.0),
                    factor: FloatRange::gain_skew_factor(-80.0, 0.0),
                },
            )
            .with_unit(" dB")
            // The value does not go down to 0 so we can do logarithmic here
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            nl_proc_type: EnumParam::new("Nonlinear Process", adaa::NLProc::HardClip),
            nl_proc_order: EnumParam::new("Antiderivative Order", AntiderivativeOrder::First),
            os_level: EnumParam::new("Oversample Factor", OversampleFactor::TwoTimes),
            pre_filter_cutoff: FloatParam::new(
                "Prefilter Cutoff Frequency",
                700.0,
                FloatRange::Skewed {
                    min: 100.,
                    max: 22000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(100.0))
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),
        }
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
            // self.peak_meter.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let num_channels = _audio_io_layout
            .main_output_channels
            .expect("Plugin was initialized without any outputs")
            .get() as usize;

        self.proc_style = self.params.nl_proc_type.value();

        self.first_order_nlprocs
            .resize_with(num_channels, || adaa::ADAAFirst::new(self.proc_style));
        self.second_order_nlprocs
            .resize_with(num_channels, || adaa::ADAASecond::new(self.proc_style));

        self.oversamplers.resize_with(num_channels, || {
            Oversample::new(self.params.os_level.value(), MAX_BLOCK_SIZE as u32)
        });

        self.oversamplers
            .iter_mut()
            .for_each(|os| os.initialize_oversample_stages());

        self.pre_filters[0].init(
            &buffer_config.sample_rate,
            &self.params.pre_filter_cutoff.value(),
            FilterOrder::First,
        );

        self.pre_filters[1].init(
            &buffer_config.sample_rate,
            &self.params.pre_filter_cutoff.value(),
            FilterOrder::First,
        );

        // Resize buffers and perform other potentially expensive initialization operations here.
        // The `reset()` function is always called right after this function. You can remove this
        // function if you do not need it.
        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
        self.first_order_nlprocs.iter_mut().for_each(|&mut x| {
            x.reset(self.params.nl_proc_type.value());
        });
        self.second_order_nlprocs.iter_mut().for_each(|&mut x| {
            x.reset(self.params.nl_proc_type.value());
        });

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
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for (_, block) in buffer.iter_blocks(MAX_BLOCK_SIZE as usize) {
            for (block_channel, (oversampler, (filter, (first_order_proc, second_order_proc)))) in
                block.into_iter().zip(
                    self.oversamplers.iter_mut().zip(
                        self.pre_filters.iter_mut().zip(
                            self.first_order_nlprocs
                                .iter_mut()
                                .zip(self.second_order_nlprocs.iter_mut()),
                        ),
                    ),
                )
            {
                if self.params.os_level.value() != oversampler.get_oversample_factor() {
                    oversampler.set_oversample_factor(self.params.os_level.value());
                };

                // set cutoff
                let param_pre_filter_cutoff: &Smoother<f32> =
                    &self.params.pre_filter_cutoff.smoothed;
                if param_pre_filter_cutoff.is_smoothing() {
                    filter.set_cutoff(param_pre_filter_cutoff.next());
                };

                filter.process_block(block_channel);

                let num_processed_samples =
                    oversampler.process_up(block_channel, &mut self.over_sample_process_buf);

                let gain = self.params.gain.smoothed.next();
                let output = self.params.output.smoothed.next();
                let order = self.params.nl_proc_order.value();
                let style = self.params.nl_proc_type.value();

                if style != self.proc_style {
                    first_order_proc.reset(style);
                    second_order_proc.reset(style);
                }

                if order != self.proc_order {
                    self.proc_order = order;
                }

                let proc =
                    |input: &f32, f_proc: &mut adaa::ADAAFirst, s_proc: &mut adaa::ADAASecond| {
                        match self.proc_order {
                            AntiderivativeOrder::First => f_proc.next_adaa(input),
                            AntiderivativeOrder::Second => s_proc.next_adaa(input),
                        }
                    };

                self.over_sample_process_buf
                    .iter_mut()
                    .take(num_processed_samples)
                    .for_each(|sample| {
                        *sample *= gain;
                        *sample = proc(sample, first_order_proc, second_order_proc);
                        *sample *= output;
                    });
                oversampler.process_down(&mut self.over_sample_process_buf, block_channel);
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
