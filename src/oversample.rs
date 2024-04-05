use core::panic;

use crate::oversample::os_filter_constants::FILTER_TAPS;
use crate::oversample::oversample_stage::OversampleStage;
use ndarray::Array1;
use nih_plug::prelude::*;
use num_traits::Float;

mod os_filter_constants;
mod oversample_stage;

const MAX_OVER_SAMPLE_FACTOR: u32 = 4;

/*
#[derive(Debug, Clone, Copy)]
pub struct OSFactorScale {
    factor: u32,
    scale: u32,
}

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
*/

#[derive(Enum, Debug, Copy, Clone, PartialEq)]
pub enum OversampleFactor {
    #[id = "2x"]
    #[name = "2x"]
    TwoTimes,
    #[id = "4x"]
    #[name = "4x"]
    FourTimes,
    #[id = "8x"]
    #[name = "8x"]
    EightTimes,
    #[id = "16x"]
    #[name = "16x"]
    SixteenTimes,
}

enum SampleRole {
    UpSample,
    DownSample,
}

#[derive(Debug)]
pub struct Oversample<T>
where
    T: Float + Copy + From<f32> + 'static,
{
    factor: OversampleFactor,
    up_stages: Vec<OversampleStage<T>>,
    down_stages: Vec<OversampleStage<T>>,
    kernel: Array1<T>,
}

impl<T> Oversample<T>
where
    T: Float + Copy + From<f32> + std::fmt::Debug + 'static,
{
    pub fn new(initial_factor: OversampleFactor, initial_buff_size: u32) -> Self {
        Oversample {
            factor: initial_factor,
            up_stages: (1..=MAX_OVER_SAMPLE_FACTOR)
                .map(|factor| {
                    OversampleStage::new(
                        (initial_buff_size * 2_u32.pow(factor as u32)) as usize,
                        SampleRole::UpSample,
                    )
                })
                .collect::<Vec<_>>(),
            down_stages: (1..=MAX_OVER_SAMPLE_FACTOR)
                .map(|factor| {
                    OversampleStage::new(
                        (initial_buff_size * 2_u32.pow(MAX_OVER_SAMPLE_FACTOR - factor)) as usize,
                        SampleRole::DownSample,
                    )
                })
                .collect::<Vec<_>>(),
            kernel: Array1::from_iter(FILTER_TAPS.into_iter().map(|x| (x as f32).into())),
        }
    }

    pub fn get_oversample_factor(&self) -> OversampleFactor {
        self.factor
    }

    pub fn set_oversample_factor(&mut self, new_factor: OversampleFactor) {
        self.factor = new_factor;
    }

    pub fn reset(&mut self) {
        for (up_stage, down_stage) in self.up_stages.iter_mut().zip(self.down_stages.iter_mut()) {
            up_stage.reset();
            down_stage.reset();
        }
    }

    pub fn process_up(&mut self, input: &[T], output: &mut [T]) -> usize {
        let mut stages = self.up_stages.iter_mut();

        let mut last_stage = match stages.next() {
            Some(stage) => {
                stage.process_up(input, &self.kernel);
                &stage.data
            }
            None => panic!(
                "There must be at least one up sample stage for processing the raw input data"
            ),
        };

        let num_remaining_stages: usize = match self.factor {
            OversampleFactor::TwoTimes => 0,
            OversampleFactor::FourTimes => 1,
            OversampleFactor::EightTimes => 2,
            OversampleFactor::SixteenTimes => 3,
        };

        for (idx, stage) in stages.enumerate() {
            if idx < num_remaining_stages {
                stage.process_up(last_stage, &self.kernel);
                last_stage = &stage.data;
            } else {
                break;
            }
        }

        for (i, out) in last_stage.iter().enumerate() {
            output[i] = *out;
        }

        last_stage.len()
    }

    pub fn process_down(&mut self, input: &[T], output: &mut [T]) {
        let num_remaining_stages: usize = match self.factor {
            OversampleFactor::TwoTimes => 1,
            OversampleFactor::FourTimes => 2,
            OversampleFactor::EightTimes => 3,
            OversampleFactor::SixteenTimes => 4,
        };

        let mut stages = self
            .down_stages
            .iter_mut()
            .rev()
            .take(num_remaining_stages)
            .rev();

        let mut last_stage = match stages.next() {
            Some(stage) => {
                stage.process_down(input, &self.kernel);
                &stage.data
            }
            None => panic!(
                "There must be at least one up sample stage for processing the raw input data"
            ),
        };

        for (idx, stage) in stages.enumerate() {
            if idx < num_remaining_stages {
                stage.process_down(last_stage, &self.kernel);
                last_stage = &mut stage.data;
            } else {
                break;
            }
        }

        for (i, out) in last_stage.iter().enumerate() {
            output[i] = *out;
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::oversample::*;

    #[test]
    fn test_create_os_2x() {
        let os = Oversample::<f32>::new(OversampleFactor::TwoTimes, 4);
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.down_stages[3].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::TwoTimes, 4);
        assert_eq!(os_64.up_stages.len(), 4);
        assert_eq!(os_64.down_stages.len(), 4);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.down_stages[3].data.len(), 4);
    }

    #[test]
    fn test_create_os_4x() {
        let os = Oversample::<f32>::new(OversampleFactor::FourTimes, 4);
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.up_stages[1].data.len(), 16);
        assert_eq!(os.down_stages[2].data.len(), 8);
        assert_eq!(os.down_stages[3].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::FourTimes, 4);
        assert_eq!(os_64.up_stages.len(), 4);
        assert_eq!(os_64.down_stages.len(), 4);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.up_stages[1].data.len(), 16);
        assert_eq!(os_64.down_stages[2].data.len(), 8);
        assert_eq!(os_64.down_stages[3].data.len(), 4);
    }

    #[test]
    fn test_create_os_8x() {
        let os = Oversample::<f32>::new(OversampleFactor::EightTimes, 4);
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.up_stages[1].data.len(), 16);
        assert_eq!(os.up_stages[2].data.len(), 32);
        assert_eq!(os.down_stages[1].data.len(), 16);
        assert_eq!(os.down_stages[2].data.len(), 8);
        assert_eq!(os.down_stages[3].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::EightTimes, 4);
        assert_eq!(os_64.up_stages.len(), 4);
        assert_eq!(os_64.down_stages.len(), 4);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.up_stages[1].data.len(), 16);
        assert_eq!(os_64.up_stages[2].data.len(), 32);
        assert_eq!(os_64.down_stages[1].data.len(), 16);
        assert_eq!(os_64.down_stages[2].data.len(), 8);
        assert_eq!(os_64.down_stages[3].data.len(), 4);
    }

    #[test]
    fn test_create_os_16x() {
        let os = Oversample::<f32>::new(OversampleFactor::SixteenTimes, 4);
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.up_stages[1].data.len(), 16);
        assert_eq!(os.up_stages[2].data.len(), 32);
        assert_eq!(os.up_stages[3].data.len(), 64);
        assert_eq!(os.down_stages[0].data.len(), 32);
        assert_eq!(os.down_stages[1].data.len(), 16);
        assert_eq!(os.down_stages[2].data.len(), 8);
        assert_eq!(os.down_stages[3].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 4);
        assert_eq!(os_64.up_stages.len(), 4);
        assert_eq!(os_64.down_stages.len(), 4);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.up_stages[1].data.len(), 16);
        assert_eq!(os_64.up_stages[2].data.len(), 32);
        assert_eq!(os_64.up_stages[3].data.len(), 64);
        assert_eq!(os_64.down_stages[0].data.len(), 32);
        assert_eq!(os_64.down_stages[1].data.len(), 16);
        assert_eq!(os_64.down_stages[2].data.len(), 8);
        assert_eq!(os_64.down_stages[3].data.len(), 4);
    }

    #[test]
    fn test_small_up_sample_2x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::TwoTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result: &mut [f64] = &mut [0.0; 8];
        os.process_up(sig, result);

        let expected_result: &[f64] = &[
            -0.012943094819578109,
            0.0,
            0.013577449569054703,
            0.0,
            -0.014268251144141814,
            0.0,
            0.015023742543533445,
            0.0,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn test_small_up_sample_4x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::FourTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        const E_RESULT: &[f64] = &[
            0.00016752370350858967,
            0.0,
            -0.00017573421718031495,
            0.0,
            8.941110287866394e-06,
            0.0,
            -1.0106587485659248e-05,
            0.0,
            0.00019614686008995804,
            0.0,
            -0.00020680688566223008,
            0.0,
            2.411980294562719e-05,
            0.0,
            -2.7654982156956038e-05,
            0.0,
        ];

        let result: &mut [f64] = &mut [0.0; E_RESULT.len()];
        os.process_up(sig, result);

        for (r, e) in result.iter().zip(E_RESULT.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn test_small_up_sample_8x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result: &mut [f64] = &mut [0.0; 32];
        os.process_up(sig, result);

        let expected_result: &[f64] = &[
            -2.168275179038566e-06,
            0.0,
            2.2745446360091485e-06,
            0.0,
            -1.1572563824815977e-07,
            0.0,
            1.3081052012924913e-07,
            0.0,
            -2.6420277269769886e-07,
            0.0,
            2.9069865914485977e-07,
            0.0,
            -1.9078742252992996e-07,
            0.0,
            2.207193743894453e-07,
            0.0,
            -2.7956397603268105e-06,
            0.0,
            2.9642240835066686e-06,
            0.0,
            -4.775282888187613e-07,
            0.0,
            5.627502326401241e-07,
            0.0,
            -9.8157074847025e-07,
            0.0,
            1.1325599486140385e-06,
            0.0,
            -9.674399702727956e-07,
            0.0,
            1.2024669215155766e-06,
            0.0,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn test_small_up_sample_16x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result: &mut [f64] = &mut [0.0; 64];
        os.process_up(sig, result);

        let expected_result: &[f64] = &[
            2.806419123723386e-08,
            0.0,
            -2.943964689522919e-08,
            0.0,
            1.497847908902133e-09,
            0.0,
            -1.6930929654312083e-09,
            0.0,
            3.419601538621766e-09,
            0.0,
            -3.7625403092361434e-09,
            0.0,
            2.4693797001877947e-09,
            0.0,
            -2.85679179124055e-09,
            0.0,
            6.744583604063339e-09,
            0.0,
            -7.483718291325725e-09,
            0.0,
            4.609434904051034e-09,
            0.0,
            -5.507656380644065e-09,
            0.0,
            9.117363447267262e-09,
            0.0,
            -1.071188441943685e-08,
            0.0,
            9.931260659680578e-09,
            0.0,
            -1.25668372078963e-08,
            0.0,
            5.24306626266722e-08,
            0.0,
            -5.954520923990405e-08,
            0.0,
            3.126451985754981e-08,
            0.0,
            -4.4823188113968115e-08,
            0.0,
            7.528493772783126e-08,
            0.0,
            -1.2626282742369094e-07,
            0.0,
            2.592588706683015e-07,
            0.0,
            -1.1034782145228508e-06,
            -2.1819776940451505e-06,
            -1.854080815038716e-06,
            0.0,
            1.8949327956055762e-06,
            2.288918725750784e-06,
            1.19976396648328e-06,
            0.0,
            -3.551339101598394e-07,
            -1.1645697175696547e-07,
            6.247181640164059e-08,
            0.0,
            -9.603776483706852e-09,
            1.31637183244899e-07,
            1.651197305146256e-07,
            0.0,
            -2.0487917826826312e-07,
            -2.6587241430623127e-07,
        ];
        assert_eq!(result.len(), expected_result.len());

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_2x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::TwoTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 7]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result: &mut [f64] = &mut [0.0; 4];
        os.process_down(sig, result);

        let expected_result: &[f64] = &[
            -0.0064715474097890545,
            0.006788724784527351,
            -0.007134125572070907,
            0.007511871271766723,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_4x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::FourTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 15]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result: &mut [f64] = &mut [0.0; 4];
        os.process_down(sig, result);
        let expected_result: &[f64] = &[
            4.188092587714742e-05,
            2.2352775719665985e-06,
            4.903671502248951e-05,
            6.029950736406798e-06,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_8x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 31]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result: &mut [f64] = &mut [0.0; 4];
        os.process_down(sig, result);

        let expected_result: &[f64] = &[
            -2.7103439737982077e-07,
            -3.302534658721244e-08,
            -3.494549700408512e-07,
            -1.2269634355878129e-07,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_16x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 63]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result: &mut [f64] = &mut [0.0; 4];
        os.process_down(sig, result);

        let expected_result: &[f64] = &[
            1.7540119523271163e-09,
            4.2153647525395777e-10,
            3.276916414167012e-09,
            -1.1588005093991977e-07,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }
}
