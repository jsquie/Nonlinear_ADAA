use nih_plug::prelude::*;
use std::sync::Arc;

mod adaa;
mod circular_buffer;
mod oversample;
// This is a shortened version of the gain example with most comments removed, check out
// https://github.com/robbert-vdh/nih-plug/blob/master/plugins/examples/gain/src/lib.rs to get
// started

const MAX_BLOCK_SIZE: u32 = 32;

pub struct NonlinearAdaa {
    params: Arc<NonlinearAdaaParams>,
    first_order_nlprocs: Vec<adaa::ADAAFirst>,
    second_order_nlprocs: Vec<adaa::ADAASecond>,
    proc_style: adaa::NLProc,
    os: Vec<oversample::Oversample<f32>>,
    os_scratch_buf: [f32; (MAX_BLOCK_SIZE * 16) as usize],
}

#[derive(Enum, Debug, PartialEq)]
enum AntiderivativeOrder {
    #[id = "first order ad"]
    #[name = "First Order Anti Derivative"]
    First,

    #[id = "second order ad"]
    #[name = "Second Order Anti Derivataive"]
    Second,
}

#[derive(Params)]
struct NonlinearAdaaParams {
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
    pub os_level: EnumParam<oversample::OversampleFactor>,
}

impl Default for NonlinearAdaa {
    fn default() -> Self {
        Self {
            params: Arc::new(NonlinearAdaaParams::default()),
            first_order_nlprocs: Vec::with_capacity(2),
            second_order_nlprocs: Vec::with_capacity(2),
            proc_style: adaa::NLProc::HardClip,
            os: Vec::with_capacity(2),
            os_scratch_buf: [0.; (MAX_BLOCK_SIZE * 16) as usize],
        }
    }
}

impl Default for NonlinearAdaaParams {
    fn default() -> Self {
        Self {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
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
                    min: util::db_to_gain(-140.0),
                    max: util::db_to_gain(0.0),
                    factor: FloatRange::gain_skew_factor(-140.0, 0.0),
                },
            )
            .with_unit(" dB")
            // The value does not go down to 0 so we can do logarithmic here
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            nl_proc_type: EnumParam::new("Nonlinear Process", adaa::NLProc::HardClip),
            nl_proc_order: EnumParam::new("Antiderivative Order", AntiderivativeOrder::First),
            os_level: EnumParam::new("Oversample Factor", oversample::OversampleFactor::TwoTimes),
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
        self.os.resize_with(num_channels, || {
            oversample::Oversample::<f32>::new(self.params.os_level.value(), MAX_BLOCK_SIZE)
        });

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

        self.os.iter_mut().for_each(|x| x.reset());
        self.os_scratch_buf = [0.0; (MAX_BLOCK_SIZE * 16) as usize];
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for (_, block) in buffer.iter_blocks(MAX_BLOCK_SIZE as usize) {
            for (block_channel, (os, (first_order_proc, second_order_proc))) in
                block.into_iter().zip(
                    self.os.iter_mut().zip(
                        self.first_order_nlprocs
                            .iter_mut()
                            .zip(self.second_order_nlprocs.iter_mut()),
                    ),
                )
            {
                let gain = self.params.gain.smoothed.next();
                let output = self.params.output.smoothed.next();
                let order = self.params.nl_proc_order.value();
                let style = self.params.nl_proc_type.value();

                os.process_up(block_channel, &self.os_scratch_buf);

                if style != self.proc_style {
                    first_order_proc.reset(style);
                    second_order_proc.reset(style);
                }

                for sample in block_channel.iter_mut() {
                    *sample *= gain;
                    *sample = match order {
                        AntiderivativeOrder::First => first_order_proc.next_adaa(sample),
                        AntiderivativeOrder::Second => second_order_proc.next_adaa(sample),
                    };
                    *sample *= output;
                    // sample = nl_proc.next_adaa(&sample);
                }
            }
        }
        /*
        for channel_samples in buffer.iter_samples() {
            // Smoothing is optionally built into the parameters themselves
            let mut first_order_processor = self.first_order_nlprocs.iter_mut();
            let mut second_order_processor = self.second_order_nlprocs.iter_mut();

            // let adaa_order = self.params.nl_proc_type;
            // let output = self.params.output.smoothed.next();
            for s in channel_samples.into_iter() {
                let mut current_sample = *s;
                current_sample *= gain;

                current_sample = match self.params.nl_proc_order.value() {
                    AntiderivativeOrder::First => match first_order_processor.next() {
                        Some(x) => x.next_adaa(&current_sample),
                        _ => panic!("Crap. There was no first order processor"),
                    },
                    AntiderivativeOrder::Second => match second_order_processor.next() {
                        Some(x) => x.next_adaa(&current_sample),
                        _ => panic!("Crap, there was no second order processor"),
                    },
                };
                current_sample *= output;

                *s = current_sample.clone();
            }
        }
        */

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
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(NonlinearAdaa);
nih_export_vst3!(NonlinearAdaa);
