use ndarray::Array1;

use crate::circular_buffer::CircularBuffer;
use nih_plug::prelude::*;
use num_traits::Float;

#[allow(dead_code)]
const MAX_OVERSAMPLE_FACTOR: u8 = 4;

#[allow(dead_code)]
const NUM_OS_FILTER_TAPS: usize = 48;
#[allow(dead_code)]
const UP_DELAY: usize = (NUM_OS_FILTER_TAPS / 2) - 1;
#[allow(dead_code)]
const DOWN_DELAY: usize = NUM_OS_FILTER_TAPS / 2;
#[allow(dead_code)]
const FILTER_TAPS: [f64; NUM_OS_FILTER_TAPS] = [
    -0.0064715474097890545,
    0.006788724784527351,
    -0.007134125572070907,
    0.007511871271766723,
    -0.007926929217098087,
    0.00838534118242672,
    -0.00889453036904902,
    0.009463720022395613,
    -0.010104514094437885,
    0.010831718180021,
    -0.011664525313602769,
    0.012628270948224513,
    -0.013757103575462731,
    0.015098181413680897,
    -0.01671851963595936,
    0.01871667093508393,
    -0.021243750540180146,
    0.024543868940610197,
    -0.0290386730354654,
    0.035524608815134716,
    -0.045708348639099484,
    0.06402724397938601,
    -0.10675158913607562,
    0.32031404953367254,
    0.32031404953367254,
    -0.10675158913607562,
    0.06402724397938601,
    -0.045708348639099484,
    0.035524608815134716,
    -0.0290386730354654,
    0.024543868940610197,
    -0.021243750540180146,
    0.01871667093508393,
    -0.01671851963595936,
    0.015098181413680897,
    -0.013757103575462731,
    0.012628270948224513,
    -0.011664525313602769,
    0.010831718180021,
    -0.010104514094437885,
    0.009463720022395613,
    -0.00889453036904902,
    0.00838534118242672,
    -0.007926929217098087,
    0.007511871271766723,
    -0.007134125572070907,
    0.006788724784527351,
    -0.0064715474097890545,
];

#[allow(dead_code)]
const FOLD_SCALE_64: f64 = 0.5031597730627207;
#[allow(dead_code)]
const FOLD_SCALE_32: f32 = FOLD_SCALE_64 as f32;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct OSFactorScale {
    factor: u32,
    scale: u32,
}

#[allow(dead_code)]
impl OSFactorScale {
    pub const TWO_TIMES: OSFactorScale = OSFactorScale {
        factor: 1,
        scale: 2,
    };
    pub const FOUR_TIMES: OSFactorScale = OSFactorScale {
        factor: 2,
        scale: 2_u32.pow(2),
    };
    pub const EIGHT_TIMES: OSFactorScale = OSFactorScale {
        factor: 3,
        scale: 2_u32.pow(3),
    };
    pub const SIXTEEN_TIMES: OSFactorScale = OSFactorScale {
        factor: 4,
        scale: 2_u32.pow(4),
    };
}

#[derive(Enum, Debug, Copy, Clone, PartialEq)]
pub enum OversampleFactor {
    #[id = "2x"]
    #[name = "Two Times"]
    TwoTimes,
    #[id = "4x"]
    #[name = "Four Times"]
    FourTimes,
    #[id = "8x"]
    #[name = "Eight Times"]
    EightTimes,
    #[id = "16x"]
    #[name = "Sixteen Times"]
    SixteenTimes,
}
#[allow(dead_code)]
enum SampleRole {
    UpSample,
    DownSample,
}

#[allow(dead_code)]
struct OversampleStage<T>
where
    T: Float + 'static,
{
    filter_buff: CircularBuffer<T>,
    delay_buff: CircularBuffer<T>,
    data: Vec<T>,
    size: usize,
}

#[allow(dead_code)]
impl OversampleStage<f32> {
    fn new(target_size: u32, role: SampleRole) -> OversampleStage<f32> {
        OversampleStage {
            filter_buff: CircularBuffer::new(NUM_OS_FILTER_TAPS),
            delay_buff: match role {
                SampleRole::UpSample => CircularBuffer::new(UP_DELAY),
                SampleRole::DownSample => CircularBuffer::new(DOWN_DELAY),
            },
            data: vec![0.0_f32; target_size as usize],
            size: target_size as usize,
        }
    }

    fn reset(&mut self) {
        self.filter_buff.reset();
        self.delay_buff.reset();
        self.data = vec![0.0_f32; self.size];
    }

    fn process_up(&mut self, input: &Vec<f32>, kernel: &Array1<f32>) {
        let mut output = self.data.iter_mut();

        input.iter().for_each(|x| {
            match output.next() {
                Some(out) => *out = self.filter_buff.convolve(*x, kernel) * 2.0_f32,
                None => (),
            }
            match output.next() {
                Some(out) => *out = self.delay_buff.delay(*x * 2.0_f32 * FOLD_SCALE_32),
                None => (),
            }
        });
    }

    fn process_down(&mut self, input: &Vec<f32>, kernel: &Array1<f32>) {
        let output = self.data.iter_mut();
        let mut input_itr = input.into_iter();

        for output_sample in output {
            let even_idx = match input_itr.next() {
                Some(input_sample) => self.filter_buff.convolve(*input_sample, kernel),
                None => 0.,
            };

            let odd_idx = match input_itr.next() {
                Some(input_sample) => self.delay_buff.delay(*input_sample) * FOLD_SCALE_32,
                None => 0.,
            };

            *output_sample = even_idx + odd_idx;
        }
    }
}

#[allow(dead_code)]
impl OversampleStage<f64> {
    fn new(target_size: u32, role: SampleRole) -> OversampleStage<f64> {
        OversampleStage {
            filter_buff: CircularBuffer::new(NUM_OS_FILTER_TAPS),
            delay_buff: match role {
                SampleRole::UpSample => CircularBuffer::new(UP_DELAY),
                SampleRole::DownSample => CircularBuffer::new(DOWN_DELAY),
            },
            data: vec![0.0_f64; target_size as usize],
            size: target_size as usize,
        }
    }

    fn reset(&mut self) {
        self.filter_buff.reset();
        self.delay_buff.reset();
        self.data = vec![0.0_f64; self.size];
    }

    fn process_up(&mut self, input: &Vec<f64>, kernel: &Array1<f64>) {
        let mut output = self.data.iter_mut();

        input.iter().for_each(|input_sample| {
            match output.next() {
                Some(output_sample) => {
                    *output_sample = self.filter_buff.convolve(*input_sample, kernel) * 2.0_f64
                }
                None => (),
            }
            match output.next() {
                Some(output_sample) => {
                    *output_sample = self
                        .delay_buff
                        .delay(*input_sample * 2.0_f64 * FOLD_SCALE_64)
                }
                None => (),
            }
        });
    }

    fn process_down(&mut self, input: &Vec<f64>, kernel: &Array1<f64>) {
        let output = self.data.iter_mut();
        let mut input_itr = input.into_iter();

        for output_sample in output {
            let even_idx = match input_itr.next() {
                Some(input_sample) => self.filter_buff.convolve(*input_sample, kernel),
                None => 0.,
            };

            let odd_idx = match input_itr.next() {
                Some(input_sample) => self.delay_buff.delay(*input_sample) * FOLD_SCALE_64,
                None => 0.,
            };

            *output_sample = even_idx + odd_idx;
        }
    }
}

#[allow(dead_code)]
impl<T> OversampleStage<T>
where
    T: Float + 'static,
{
    fn get_processed_data(&self) -> &Vec<T> {
        &self.data
    }
}

#[allow(dead_code)]
struct Oversample<T>
where
    T: Float + 'static,
{
    buff_size: u32,
    factor: OversampleFactor,
    factor_scale: OSFactorScale,
    up_stages: Vec<OversampleStage<T>>,
    down_stages: Vec<OversampleStage<T>>,
    kernel: Array1<T>,
}

#[allow(dead_code)]
impl Oversample<f32> {
    fn new(initial_factor: OversampleFactor, initial_buff_size: u32) -> Oversample<f32> {
        let new_factor: OSFactorScale = match initial_factor {
            OversampleFactor::TwoTimes => OSFactorScale::TWO_TIMES,
            OversampleFactor::FourTimes => OSFactorScale::FOUR_TIMES,
            OversampleFactor::EightTimes => OSFactorScale::EIGHT_TIMES,
            OversampleFactor::SixteenTimes => OSFactorScale::SIXTEEN_TIMES,
        };

        Oversample {
            buff_size: initial_buff_size,
            factor: initial_factor,
            factor_scale: new_factor,
            up_stages: (1..=new_factor.factor)
                .into_iter()
                .map(|factor| {
                    OversampleStage::<f32>::new(
                        initial_buff_size * 2_u32.pow(factor),
                        SampleRole::UpSample,
                    )
                })
                .collect::<Vec<OversampleStage<f32>>>(),
            down_stages: (1..=new_factor.factor)
                .into_iter()
                .map(|factor| {
                    OversampleStage::<f32>::new(
                        initial_buff_size * 2_u32.pow(new_factor.factor - factor),
                        SampleRole::DownSample,
                    )
                })
                .collect::<Vec<OversampleStage<f32>>>(),
            kernel: Array1::<f32>::from_iter(FILTER_TAPS.into_iter().map(|x| x as f32)),
        }
    }

    fn reset(&mut self) {
        self.up_stages.iter_mut().for_each(|stage| stage.reset());
        self.down_stages.iter_mut().for_each(|stage| stage.reset());
    }

    fn process_up(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let stages = self.up_stages.iter_mut();
        let mut last_stage = input;
        for stage in stages {
            stage.process_up(last_stage, &self.kernel);
            last_stage = stage.get_processed_data();
        }
        last_stage.to_owned()
    }

    fn process_down(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let stages = self.down_stages.iter_mut();
        let mut last_stage = input;
        for stage in stages {
            stage.process_down(last_stage, &self.kernel);
            last_stage = stage.get_processed_data();
        }
        last_stage.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use crate::oversample::OSFactorScale;
    use crate::oversample::Oversample;
    use crate::oversample::OversampleFactor;
    use crate::oversample::SampleRole;
    use crate::oversample::FILTER_TAPS;
    use ndarray::Array1;

    use super::OversampleStage;

    #[test]
    fn test_create_2x_os() {
        let os = Oversample::<f32>::new(OversampleFactor::TwoTimes, 8);
        assert_eq!(os.factor_scale.factor, 1);
        assert_eq!(os.factor_scale.scale, 2);
        assert_eq!(os.factor, OversampleFactor::TwoTimes);
        assert_eq!(
            os.kernel.into_raw_vec(),
            get_kern().iter().map(|x| *x as f32).collect::<Vec<f32>>()
        );
        assert_eq!(os.up_stages.len(), 1);
        assert_eq!(os.up_stages[0].size, 16);
        assert_eq!(os.down_stages.len(), 1);
        assert_eq!(os.down_stages[0].size, 8);
    }

    #[test]
    fn test_create_4x_os() {
        let os = Oversample::<f32>::new(OversampleFactor::FourTimes, 8);
        assert_eq!(os.factor_scale.factor, 2);
        assert_eq!(os.factor_scale.scale, 4);
        assert_eq!(os.factor, OversampleFactor::FourTimes);
        assert_eq!(
            os.kernel.into_raw_vec(),
            get_kern().iter().map(|x| *x as f32).collect::<Vec<f32>>()
        );
        assert_eq!(os.up_stages.len(), 2);
        assert_eq!(os.up_stages[0].size, 16);
        assert_eq!(os.up_stages[1].size, 32);
        assert_eq!(os.down_stages.len(), 2);
        assert_eq!(os.down_stages[0].size, 16);
        assert_eq!(os.down_stages[1].size, 8);
    }

    #[test]
    fn test_create_8x_os() {
        let os = Oversample::<f32>::new(OversampleFactor::EightTimes, 8);
        assert_eq!(os.factor_scale.factor, 3);
        assert_eq!(os.factor_scale.scale, 8);
        assert_eq!(os.factor, OversampleFactor::EightTimes);
        assert_eq!(
            os.kernel.into_raw_vec(),
            get_kern().iter().map(|x| *x as f32).collect::<Vec<f32>>()
        );
        assert_eq!(os.up_stages.len(), 3);
        assert_eq!(os.up_stages[0].size, 16);
        assert_eq!(os.up_stages[1].size, 32);
        assert_eq!(os.up_stages[2].size, 64);
        assert_eq!(os.down_stages.len(), 3);
        assert_eq!(os.down_stages[0].size, 32);
        assert_eq!(os.down_stages[1].size, 16);
        assert_eq!(os.down_stages[2].size, 8);
    }

    #[test]
    fn test_create_16x_os() {
        let os = Oversample::<f32>::new(OversampleFactor::SixteenTimes, 8);
        assert_eq!(os.factor_scale.factor, 4);
        assert_eq!(os.factor_scale.scale, 16);
        assert_eq!(os.factor, OversampleFactor::SixteenTimes);
        assert_eq!(
            os.kernel.into_raw_vec(),
            get_kern().iter().map(|x| *x as f32).collect::<Vec<f32>>()
        );
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.up_stages[0].size, 16);
        assert_eq!(os.up_stages[1].size, 32);
        assert_eq!(os.up_stages[2].size, 64);
        assert_eq!(os.up_stages[3].size, 128);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(os.down_stages[0].size, 64);
        assert_eq!(os.down_stages[1].size, 32);
        assert_eq!(os.down_stages[2].size, 16);
        assert_eq!(os.down_stages[3].size, 8);
    }

    #[test]
    fn test_up_sample_2x() {
        unimplemented!()
    }

    fn get_kern() -> Array1<f64> {
        Array1::<f64>::from_iter(FILTER_TAPS.into_iter())
    }

    #[test]
    fn test_2x_factor_scale() {
        let two = OSFactorScale::TWO_TIMES;
        assert_eq!(two.factor, 1);
        assert_eq!(two.scale, 2);
    }

    #[test]
    fn test_4x_factor_scale() {
        let two = OSFactorScale::FOUR_TIMES;
        assert_eq!(two.factor, 2);
        assert_eq!(two.scale, 4);
    }

    #[test]
    fn test_8x_factor_scale() {
        let two = OSFactorScale::EIGHT_TIMES;
        assert_eq!(two.factor, 3);
        assert_eq!(two.scale, 8);
    }

    #[test]
    fn test_16x_factor_scale() {
        let two = OSFactorScale::SIXTEEN_TIMES;
        assert_eq!(two.factor, 4);
        assert_eq!(two.scale, 16);
    }

    // tests for OversampleStage
    #[test]
    fn test_os_up_stage_32() {
        let os_stage: OversampleStage<f32> = OversampleStage::<f32>::new(2, SampleRole::UpSample);
        assert_eq!(os_stage.size, 2);
        assert_eq!(os_stage.data, vec![0., 0.]);
    }

    #[test]
    fn test_os_down_stage_32() {
        let os_stage: OversampleStage<f32> = OversampleStage::<f32>::new(2, SampleRole::DownSample);
        assert_eq!(os_stage.size, 2);
        assert_eq!(os_stage.data, vec![0., 0.]);
        // assert_eq!(os_stage.data, vec![]);
    }

    #[test]
    fn test_os_stage_64() {
        let os_stage: OversampleStage<f64> = OversampleStage::<f64>::new(2, SampleRole::UpSample);
        assert_eq!(os_stage.size, 2);
        assert_eq!(os_stage.data, vec![0., 0.]);
    }

    #[test]
    fn test_os_stage_simple_up() {
        let mut os_stage: OversampleStage<f64> =
            OversampleStage::<f64>::new(12, SampleRole::UpSample);
        let signal: Vec<f64> = vec![0., 1., 2., 3., 4., 0.];

        os_stage.process_up(&signal, &Array1::from_iter(FILTER_TAPS.into_iter()));

        let expected: Vec<f64> = vec![
            0.0,
            0.0,
            -0.012943094819578109,
            0.0,
            -0.012308740070101515,
            0.0,
            -0.025942636464766737,
            0.0,
            -0.024552790315898507,
            0.0,
            0.02569867149666408,
            0.0,
        ];

        assert_eq!(
            *os_stage
                .get_processed_data()
                .into_iter()
                .map(|x| *x as f32)
                .collect::<Vec<_>>(),
            expected.iter().map(|x| *x as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_os_stage_rewrite() {
        let mut os_stage: OversampleStage<f64> =
            OversampleStage::<f64>::new(10, SampleRole::UpSample);
        let first_signal: Vec<f64> = vec![0., 1., 2., 3., 4.];
        let second_signal: Vec<f64> = vec![5., 6., 7., 8., 9.];

        os_stage.process_up(&first_signal, &Array1::from_iter(FILTER_TAPS.into_iter()));

        let first_result = os_stage.get_processed_data().clone();

        os_stage.process_up(&second_signal, &Array1::from_iter(FILTER_TAPS.into_iter()));

        let second_result = os_stage.get_processed_data().clone();

        let expected_result_1: Vec<f64> = vec![
            0.0,
            0.0,
            -0.012943094819578109,
            0.0,
            -0.012308740070101515,
            0.0,
            -0.025942636464766737,
            0.0,
            -0.024552790315898507,
            0.0,
        ];
        let expected_result_2: Vec<f64> = vec![
            -0.03901680260122647,
            0.0,
            -0.03671013252170096,
            0.0,
            -0.05219252318027351,
            0.0,
            -0.048747473794054835,
            0.0,
            -0.06551145259671194,
            0.0,
        ];

        assert_eq!(
            first_result
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            expected_result_1
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        );
        assert_eq!(
            second_result
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            expected_result_2
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        );
    }

    #[test]
    fn test_delay_part_only() {
        let mut os_stage = OversampleStage::<f64>::new(64, SampleRole::UpSample);
        let signal = (0..64).into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        // let kern = Array1::<f64>::from_iter(FILTER_TAPS.into_iter());
        os_stage.process_up(&signal, &get_kern());

        let result = os_stage
            .get_processed_data()
            .into_iter()
            .skip(1)
            .step_by(2)
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        let expected_result = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0063195461254415,
            2.012639092250883,
            3.0189586383763247,
            4.025278184501766,
            5.031597730627207,
            6.037917276752649,
            7.044236822878091,
            8.050556369003532,
        ]
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

        dbg!(result.len());
        dbg!(expected_result.len());
        assert_eq!(result, expected_result);
    }

    #[test]
    // #[ignore = "Fix delay first"]
    fn test_os_delay_amt() {
        let mut os_stage = OversampleStage::<f64>::new(64, SampleRole::UpSample);
        let signal = (0..64).into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        // let kern = Array1::<f64>::from_iter(FILTER_TAPS.into_iter());

        os_stage.process_up(&signal, &get_kern());

        let expected_result = vec![
            0.0,
            0.0,
            -0.012943094819578109,
            0.0,
            -0.012308740070101515,
            0.0,
            -0.025942636464766737,
            0.0,
            -0.024552790315898514,
            0.0,
            -0.03901680260122646,
            0.0,
            -0.03671013252170097,
            0.0,
            -0.0521925231802735,
            0.0,
            -0.048747473794054835,
            0.0,
            -0.06551145259671193,
            0.0,
            -0.06061199503932703,
            0.0,
            -0.07904158810914763,
            0.0,
            -0.07221463928251924,
            0.0,
            -0.09290189760681633,
            0.0,
            -0.0833927931037516,
            0.0,
            -0.10732072787260563,
            0.0,
            -0.09381532077129168,
            0.0,
            -0.12279741475033809,
            0.0,
            -0.1026917708481641,
            0.0,
            -0.14066347301692092,
            0.0,
            -0.1075859575554084,
            0.0,
            -0.1659251393720947,
            0.0,
            -0.09620983323000902,
            0.0,
            -0.2399977053600746,
            0.0,
            0.25684252157720494,
            1.0063195461254415,
            1.3943108475818295,
            2.012639092250883,
            2.3182759953143024,
            3.0189586383763247,
            3.370295631005548,
            4.025278184501766,
            4.330898569418594,
            5.031597730627207,
            5.362550725461909,
            6.037917276752649,
            6.336125535434294,
            7.044236822878091,
            7.3587880832879025,
            8.050556369003532,
        ];

        assert_eq!(
            os_stage
                .get_processed_data()
                .into_iter()
                .map(|x| *x as f32)
                .collect::<Vec<f32>>(),
            expected_result
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        );
    }

    #[test]
    fn test_os_stage_2multi_stages() {
        let mut os_stage_0 = OversampleStage::<f64>::new(32, SampleRole::UpSample);
        let mut os_stage_1 = OversampleStage::<f64>::new(64, SampleRole::UpSample);

        let signal = (0..16).into_iter().map(|x| x as f64).collect::<Vec<f64>>();

        // let kern = Array1::<f64>::from_iter(FILTER_TAPS.into_iter());

        os_stage_0.process_up(&signal, &get_kern());

        os_stage_1.process_up(os_stage_0.get_processed_data(), &get_kern());

        let expected_result = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.00016752370350858967,
            0.0,
            -0.00017573421718031495,
            0.0,
            0.00034398851730504574,
            0.0,
            -0.00036157502184628915,
            0.0,
            0.0007166001911914599,
            0.0,
            -0.0007542227121744935,
            0.0,
            0.001113331668023501,
            0.0,
            -0.0011745253846596538,
            0.0,
            0.0017471427328399991,
            0.0,
            -0.0018470640365824877,
            0.0,
            0.0024332936513991343,
            0.0,
            -0.0025809505377502027,
            0.0,
            0.0034222057938664723,
            0.0,
            -0.003642942051680894,
            0.0,
            0.0045245578244305145,
            0.0,
            -0.0048434371316731055,
            0.0,
            0.006060517398320655,
            0.0,
            -0.006536815381885378,
            0.0,
            0.007890296954359667,
            0.0,
            -0.008629609396434939,
            0.0,
            0.010599624452624109,
            0.0,
            -0.011965650831626738,
            0.0,
            0.015144667417428146,
            0.0,
            -0.022284086533495236,
            -0.013024889304296395,
            0.009480350784677644,
            0.0,
            -0.022523426376792968,
            -0.012386525720720592,
            0.012861844112273922,
            0.0,
            -0.0321551705756773,
            -0.02610658215252139,
            0.005823331570998718,
            0.0,
            -0.03116099320279679,
            -0.024707952806808122,
        ];

        assert_eq!(
            iter_collect_32f(os_stage_1.get_processed_data().to_vec()),
            iter_collect_32f(expected_result)
        );
    }

    fn range_to_float_vec(low: u32, high: u32) -> Vec<f64> {
        (low..high)
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<f64>>()
    }

    fn iter_collect_32f(inp: Vec<f64>) -> Vec<f32> {
        inp.into_iter().map(|x| x as f32).collect::<Vec<f32>>()
    }

    #[test]
    fn test_multi_os_stage_3_stages() {
        let mut os_stage_0 = OversampleStage::<f64>::new(64, SampleRole::UpSample);
        let mut os_stage_1 = OversampleStage::<f64>::new(128, SampleRole::UpSample);
        let mut os_stage_2 = OversampleStage::<f64>::new(256, SampleRole::UpSample);

        let sig = range_to_float_vec(0, 32);

        os_stage_0.process_up(&sig, &get_kern());
        os_stage_1.process_up(&os_stage_0.get_processed_data(), &get_kern());
        os_stage_2.process_up(&os_stage_1.get_processed_data(), &get_kern());

        let result = iter_collect_32f(os_stage_2.get_processed_data().to_vec());

        let expected_result = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.168275179038566e-06,
            0.0,
            2.2745446360091485e-06,
            0.0,
            -1.1572563824815977e-07,
            0.0,
            1.3081052012924913e-07,
            0.0,
            -4.600753130774831e-06,
            0.0,
            4.839787931163157e-06,
            0.0,
            -4.222386990262499e-07,
            0.0,
            4.823404146479441e-07,
            0.0,
            -9.828870842837906e-06,
            0.0,
            1.0369255309823835e-05,
            0.0,
            -1.2062800486231001e-06,
            0.0,
            1.3966205418067599e-06,
            0.0,
            -1.603855930337123e-05,
            0.0,
            1.703128263709855e-05,
            0.0,
            -2.9577613684927457e-06,
            0.0,
            3.5133675904811584e-06,
            0.0,
            -2.6841494318174964e-05,
            0.0,
            2.8895932604369422e-05,
            0.0,
            -7.498137782589878e-06,
            0.0,
            9.539339161530052e-06,
            0.0,
            -4.419946541457185e-05,
            0.0,
            5.138341609411515e-05,
            0.0,
            -3.287371929833348e-05,
            0.0,
            0.00010183952757235151,
            0.00016858237728001698,
            7.743787711557534e-05,
            0.0,
            -6.770393976354534e-05,
            -0.0001768447776716043,
            -0.0001534780853359649,
            0.0,
            0.00022521010652716143,
            0.0003461623686067772,
            0.00018814765505041056,
            0.0,
            -0.00017581691144758171,
            -0.0003638600118746543,
            -0.00030785114022194765,
            0.0,
            0.00045384314060094446,
            0.0007211287791531945,
            0.00045769126029043545,
            0.0,
            -0.00044661646182827625,
            -0.0007589890573929357,
            -0.0005771896560783624,
            0.0,
            0.0007268381691119248,
            0.0011203674188524904,
            0.0007234622025322155,
            0.0,
            -0.000714452387581653,
            -0.0011819478520035125,
            -0.0008949959897277493,
            0.0,
            0.0011285257442734502,
            0.0017581838819279113,
            0.001176270977234184,
            0.0,
            -0.0011747146924092255,
            -0.001858736642958315,
            -0.001351909974348114,
            0.0,
            0.0015916326284975747,
            0.002448670962865895,
            0.0016299867941138827,
            0.0,
            -0.0016379505740922525,
            -0.002597260973720998,
            -0.0018162186594784491,
            0.0,
            0.002319603275397027,
            0.0034438325812315646,
            0.0023637783199316593,
            0.0,
            -0.0023925293519215774,
            -0.003665963792008802,
            -0.002569308500336552,
            0.0,
            0.0030804837715714506,
            0.00455315097629923,
            0.0031256426545101957,
            0.0,
            -0.003182117519021019,
            -0.004874045456032389,
            -0.0033790311944948137,
            0.0,
            0.00419202223192115,
            0.006098817117563383,
            0.004269097095200837,
            0.0,
            -0.0043766739617092515,
            -0.0065781250882046985,
            -0.004603892520434708,
            0.0,
            0.005448741652305587,
            0.007940160049906175,
            0.0055719864042634865,
            0.0,
            -0.0057655857976526715,
            -0.008684144611060252,
            -0.006086152931693518,
            0.0,
            0.007343113659240696,
            0.010666609268264824,
            0.007619745267898757,
            0.0,
            -0.008028271618957595,
            -0.01204126831397813,
            -0.008612409674221553,
            0.0,
            0.010251987547087489,
            0.015240374841727055,
            0.011241386674610885,
            0.0,
            -0.013303260682079793,
            -0.02242491184620699,
            -0.021841087243564723,
            -0.013107200693033666,
            0.00030086034669223125,
            0.00954026229874678,
            0.009285563432270381,
            0.0,
            -0.013066009662958365,
            -0.022665764208684096,
            -0.022162225978720338,
            -0.012464802941346652,
            0.002525006334741209,
            0.012943125129399675,
            0.012101311103915649,
            0.0,
            -0.017699643971222104,
            -0.03235837665930173,
            -0.035250202505406544,
            -0.02627156390261188,
            -0.008737609455121592,
            0.005860132383465384,
            0.009286105061537297,
            0.0,
            -0.016655080938317468,
            -0.031357916536656434,
            -0.034389858886320186,
            -0.024864095854235976,
            -0.006278900560748135,
            0.009166733419251241,
            0.011951623454704068,
            0.0,
            -0.021174609015832385,
            -0.041063091041367225,
            -0.04785005032285093,
            -0.03951149776954372,
            -0.018654101036006676,
            0.0010352651612381223,
            0.00849693226210749,
            0.0,
            -0.019501228022367256,
            -0.039035404077474495,
            -0.045865316409030694,
            -0.037175581353384585,
            -0.015645558511048333,
            0.004585217082171151,
            0.011256880112237808,
            0.0,
            -0.024139252270758905,
            -0.04905199791043856,
            -0.05974036497676024,
            -0.0528542736907768,
            -0.029273124500382962,
            -0.004550838981198677,
            0.00755273494837774,
            0.0,
            -0.022230184137303502,
            -0.04600434770507821,
            -0.05617363149644718,
            -0.0493655444237797,
            -0.02583612426111755,
            -0.0005339650882134591,
            0.010930266185229107,
            0.0,
            -0.02752893520224504,
            -0.056552898156304286,
            -0.07051193877339935,
            -0.0663420741983913,
            -0.041278058966697236,
            -0.01069826645985343,
            0.007257219400293776,
            0.0,
            -0.025690371668119834,
            -0.052437214490318636,
            -0.06488236341017546,
            -0.06138049627697284,
            -0.037477014481193054,
            -0.006120886073485944,
            0.011575738801527579,
            0.0,
            -0.032027552494536864,
            -0.06361892923824299,
            -0.07949898339358663,
            -0.08004375869020104,
            -0.05541427286227391,
            -0.016931877666705154,
            0.009186875708785593,
            0.0,
            -0.03180825984187419,
            -0.05881001123690798,
            -0.07003544629903893,
            -0.07313025078200468,
            -0.054677692729628545,
            -0.015798076540037502,
            0.013849765481281386,
            0.0,
            -0.04436099984382034,
            -0.07955302553634001,
            -0.09024295645337767,
            -0.09407980345274924,
            -0.07831990539829856,
            -0.03641100700346595,
            0.005022338313647588,
            0.0,
            -0.04343344543694041,
            -0.07750806233121903,
            -0.08309540290905956,
            -0.08445013273874283,
            -0.07309909209378537,
            -0.034018223018011634,
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_os_stage_simple_down() {
        let mut os_stage = OversampleStage::<f64>::new(16, SampleRole::DownSample);
        let sig_to_down_sample = range_to_float_vec(0, 32);

        os_stage.process_down(&sig_to_down_sample, &get_kern());

        let result = iter_collect_32f(os_stage.get_processed_data().to_vec());
        let expected_result = vec![
            0.0,
            -0.012943094819578109,
            -0.012308740070101515,
            -0.025942636464766737,
            -0.024552790315898507,
            -0.03901680260122647,
            -0.03671013252170096,
            -0.05219252318027351,
            -0.048747473794054835,
            -0.06551145259671194,
            -0.06061199503932704,
            -0.07904158810914763,
            -0.07221463928251926,
            -0.09290189760681632,
            -0.08339279310375158,
            -0.1073207278726056,
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_os_stage_2xmulti_stage_down() {
        let mut os_stage_0 = OversampleStage::<f64>::new(32, SampleRole::DownSample);
        let mut os_stage_1 = OversampleStage::<f64>::new(16, SampleRole::DownSample);
        let sig_to_down_sample = range_to_float_vec(0, 64);

        os_stage_0.process_down(&sig_to_down_sample, &get_kern());
        os_stage_1.process_down(&os_stage_0.get_processed_data(), &get_kern());

        let result = iter_collect_32f(os_stage_1.get_processed_data().to_vec());
        let expected_result = vec![
            0.0,
            7.965659491843221e-05,
            7.533389779174369e-05,
            0.00015870132418106852,
            0.0001489576196064275,
            0.0002363480222189285,
            0.00021928636423278896,
            0.00031025949915092433,
            0.00028136823275351535,
            0.00037039114154141257,
            0.000307084565199508,
            0.0003019943385932705,
            -0.005236423591206938,
            -0.025788704078868893,
            -0.030293994192632,
            -0.05159167513433142,
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_os_stage_3xmulti_stage_down() {
        let mut os_stage_0 = OversampleStage::<f64>::new(2 << 6, SampleRole::DownSample);
        let mut os_stage_1 = OversampleStage::<f64>::new(2 << 5, SampleRole::DownSample);
        let mut os_stage_2 = OversampleStage::<f64>::new(2 << 4, SampleRole::DownSample);
        let sig_to_down_sample = range_to_float_vec(0, 2 << 7);

        os_stage_0.process_down(&sig_to_down_sample, &get_kern());
        os_stage_1.process_down(&os_stage_0.get_processed_data(), &get_kern());
        os_stage_2.process_down(&os_stage_1.get_processed_data(), &get_kern());

        let result = iter_collect_32f(os_stage_2.get_processed_data().to_vec());
        let expected_result = vec![
            0.0,
            -4.875268911234724e-07,
            -4.5256519827845804e-07,
            -9.453313050662682e-07,
            -8.289969026739216e-07,
            -1.1198132979211963e-06,
            3.5063334690882735e-05,
            0.00015926393646145933,
            0.0001903666096670579,
            0.0003185641294278938,
            0.00034389220910052877,
            0.00047702526501829784,
            0.0005501579792281553,
            0.0007942506274646963,
            0.0008504651284409361,
            0.001095692231682292,
            0.001117519405783503,
            0.0013341789720759344,
            -0.017124054461247082,
            -0.05046064470923419,
            -0.06774502040101185,
            -0.10177647451657862,
            -0.11821598111347546,
            -0.15313459643039745,
            -0.16848581292643453,
            -0.20426753539177667,
            -0.21844681190923834,
            -0.255505914342946,
            -0.26829678687709246,
            -0.30752825120958716,
            -0.3392359176906097,
            -0.4071823869518219,
        ];

        assert_eq!(result, expected_result);
    }
}
