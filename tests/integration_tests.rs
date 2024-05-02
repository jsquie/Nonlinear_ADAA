use core::fmt;

use adaa_nl::{ADAAFirst, ADAASecond, NLProc, NextAdaa};
use oversampler::{Oversample, OversampleFactor};
// use std::fmt;

const ERR_TOL: f32 = 1e-5;
const INPUT_LINSPACE: [f32; 50] = [
    -2., -1.92, -1.84, -1.76, -1.68, -1.6, -1.52, -1.44, -1.36, -1.28, -1.2, -1.12, -1.04, -0.96,
    -0.88, -0.8, -0.72, -0.64, -0.56, -0.48, -0.4, -0.32, -0.24, -0.16, -0.08, 0., 0.08, 0.16,
    0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.04, 1.12, 1.2, 1.28, 1.36, 1.44,
    1.52, 1.6, 1.68, 1.76, 1.84, 1.92,
];

#[allow(dead_code)]
struct TotalResult {
    incorrect_results: Vec<Option<ADAAResult>>,
    num_incorrect_results: Option<usize>,
    num_total_tests: usize,
    perc_correct_results: Option<f32>,
    avg_difference: Option<f32>,
}

impl TotalResult {
    pub fn new(results: Vec<Option<ADAAResult>>) -> Self {
        TotalResult {
            incorrect_results: results
                .clone()
                .into_iter()
                .filter(|v| v.is_some())
                .collect(),
            num_incorrect_results: None,
            num_total_tests: results.len(),
            perc_correct_results: None,
            avg_difference: None,
        }
    }

    pub fn initialize(&mut self) {
        self.num_incorrect_results = if self.incorrect_results.len() > 0 {
            self.avg_difference = Some(
                self.incorrect_results
                    .clone()
                    .into_iter()
                    .fold(0.0, |acc, v| acc + v.unwrap().difference)
                    / self.incorrect_results.len() as f32,
            );
            Some(self.incorrect_results.len())
        } else {
            None
        };
        self.perc_correct_results = Some(
            1. - (self.num_incorrect_results.unwrap_or(0) as f32 / self.num_total_tests as f32),
        );
    }
}

impl fmt::Debug for TotalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TotalResult: \n\tnum_incorrect: {}\n\tperc_correct: {}%\n\tavg difference: {}\n\tincorrect: {:?}",
            self.num_incorrect_results.unwrap_or(0),
            self.perc_correct_results.unwrap_or(100.0),
            self.avg_difference.unwrap_or(0.0),
            self.incorrect_results
                .clone()
                .into_iter()
                .map(|v| v.unwrap())
                .collect::<Vec<_>>()
        )
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ADAAResult {
    my_result: f32,
    expected_result: f32,
    difference: f32,
}

fn check_results(result: &[f32], expected_result: &[f32]) {
    let results: Vec<Option<ADAAResult>> = result
        .into_iter()
        .zip(expected_result.iter())
        .map(|(r, e)| {
            let res = (r - e).abs() < ERR_TOL;
            if res {
                None
            } else {
                Some(ADAAResult {
                    my_result: *r,
                    expected_result: *e,
                    difference: (r - e).abs(),
                })
            }
        })
        .collect();

    let mut total = TotalResult::new(results.clone());
    total.initialize();

    assert!(total.num_incorrect_results.unwrap_or(0) == 0, "{:?}", total);
}

#[test]
fn test_2x_tanh_ad1() {
    let mut os = Oversample::new(OversampleFactor::TwoTimes, 50);
    os.initialize_oversample_stages();
    let mut ad = ADAAFirst::new(NLProc::Tanh);

    let mut input = INPUT_LINSPACE.into_iter().collect::<Vec<f32>>();
    let mut output = vec![0.0_f32; 100];
    let mut result = vec![0.0_f32; 50];

    os.process_up(&mut input, &mut output);

    output.iter_mut().for_each(|v| *v = ad.next_adaa(v));

    os.process_down(&mut output, &mut result);

    let expected_result = [
        -8.37524987e-05,
        9.53130214e-05,
        -1.85184844e-04,
        2.10349100e-04,
        -3.08702555e-04,
        3.50218482e-04,
        -4.60270240e-04,
        5.21931183e-04,
        -6.48190026e-04,
        7.35405078e-04,
        -8.84426438e-04,
        1.00523661e-03,
        -1.18701881e-03,
        1.35406144e-03,
        -1.58488604e-03,
        1.81963923e-03,
        -2.12866802e-03,
        2.47221911e-03,
        -2.92018602e-03,
        3.46842130e-03,
        -4.22168631e-03,
        5.30953197e-03,
        -7.25221869e-03,
        1.36083531e-02,
        5.93171609e-03,
        1.16999918e-02,
        6.21446411e-03,
        1.05115703e-02,
        6.16982060e-03,
        9.45500925e-03,
        6.03230605e-03,
        8.39456926e-03,
        5.88347616e-03,
        7.25222845e-03,
        5.78284952e-03,
        5.94749520e-03,
        5.80803752e-03,
        4.36126025e-03,
        6.10102565e-03,
        2.26990644e-03,
        6.98031132e-03,
        -8.50832745e-04,
        9.32679170e-03,
        -6.64347376e-03,
        1.65802841e-02,
        -2.38267832e-02,
        6.16836944e-02,
        -8.65570774e-01,
        -9.69691757e-01,
        -9.62953779e-01,
    ];

    check_results(&result, &expected_result)
}

#[test]
fn test_2x_hc_ad1() {
    let mut os = Oversample::new(oversampler::OversampleFactor::TwoTimes, 50);
    os.initialize_oversample_stages();
    let mut ad = ADAAFirst::new(NLProc::HardClip);

    let mut input = INPUT_LINSPACE.into_iter().collect::<Vec<f32>>();
    let mut output = vec![0.0_f32; 100];
    let mut result = vec![0.0_f32; 50];

    os.process_up(&mut input, &mut output);

    output.iter_mut().for_each(|v| *v = ad.next_adaa(v));

    os.process_down(&mut output, &mut result);

    let expected_result = [
        -8.37618518e-05,
        9.53228395e-05,
        -1.85204951e-04,
        2.10370295e-04,
        -3.08735446e-04,
        3.50253378e-04,
        -4.60318953e-04,
        5.21983359e-04,
        -6.48259288e-04,
        7.35480276e-04,
        -8.84523991e-04,
        1.00534458e-03,
        -1.18715822e-03,
        1.35422011e-03,
        -1.58509346e-03,
        1.81988561e-03,
        -2.12900183e-03,
        2.47264524e-03,
        -2.92081231e-03,
        3.46934537e-03,
        -4.22336087e-03,
        5.31324166e-03,
        -7.26778300e-03,
        1.40675297e-02,
        5.63224570e-03,
        1.23404752e-02,
        5.83425852e-03,
        1.13250231e-02,
        5.72864590e-03,
        1.04702209e-02,
        5.53523990e-03,
        9.65903050e-03,
        5.32998911e-03,
        8.83287005e-03,
        5.16595443e-03,
        7.93637196e-03,
        5.10789400e-03,
        6.48343541e-03,
        4.91195684e-03,
        4.46281948e-03,
        5.17844915e-03,
        1.73209346e-03,
        6.58673731e-03,
        -2.96279772e-03,
        1.17281989e-02,
        -1.59614215e-02,
        4.15053490e-02,
        -9.45509002e-01,
        -1.00071620e+00,
        -1.01696761e+00,
    ];

    check_results(&result, &expected_result)
}

#[test]
fn test_2x_tanh_ad2() {
    let mut os = Oversample::new(oversampler::OversampleFactor::TwoTimes, 50);
    os.initialize_oversample_stages();
    let mut ad = ADAASecond::new(NLProc::Tanh);

    let mut input = INPUT_LINSPACE.into_iter().collect::<Vec<f32>>();
    let mut output = vec![0.0_f32; 100];
    let mut result = vec![0.0_f32; 50];

    os.process_up(&mut input, &mut output);

    output.iter_mut().for_each(|v| *v = ad.next_adaa(v));

    os.process_down(&mut output, &mut result);

    let expected_result = [
        -5.58374931e-05,
        7.70683551e-06,
        -5.99172866e-05,
        1.67758650e-05,
        -6.55720877e-05,
        2.76769508e-05,
        -7.33715585e-05,
        4.11071171e-05,
        -8.41772690e-05,
        5.81436996e-05,
        -9.93539731e-05,
        8.05416813e-05,
        -1.21197620e-04,
        1.11366172e-04,
        -1.53898741e-04,
        1.56513526e-04,
        -2.06049273e-04,
        2.29065492e-04,
        -2.98717963e-04,
        3.65601375e-04,
        -5.02464986e-04,
        7.26022224e-04,
        -1.29967111e-03,
        4.38302786e-03,
        8.49418111e-03,
        5.93044431e-03,
        7.81364818e-03,
        5.61934724e-03,
        7.41330795e-03,
        5.17335600e-03,
        7.05642564e-03,
        4.65823760e-03,
        6.71029039e-03,
        4.07642508e-03,
        6.36686737e-03,
        3.41476611e-03,
        6.02775330e-03,
        2.64963938e-03,
        5.70574342e-03,
        1.74342814e-03,
        5.43529665e-03,
        6.30811793e-04,
        5.30128306e-03,
        -8.27704515e-04,
        5.54242002e-03,
        -3.13759280e-03,
        8.75957577e-03,
        -6.68836819e-01,
        -1.00109735e+00,
        -9.43083926e-01,
    ];

    check_results(&result, &expected_result)
}

const OS_2X_PROC_UP: &[f32] = &[
    0.02588619,
    0.,
    -0.00230416,
    0.,
    0.02628309,
    0.,
    -0.0048551,
    0.,
    0.0269638,
    0.,
    -0.00773468,
    0.,
    0.02802797,
    0.,
    -0.0110655,
    0.,
    0.02962816,
    0.,
    -0.01503983,
    0.,
    0.03201023,
    0.,
    -0.01997722,
    0.,
    0.03559735,
    0.,
    -0.02645036,
    0.,
    0.04118445,
    0.,
    -0.03559647,
    0.,
    0.05045896,
    0.,
    -0.05003508,
    0.,
    0.06772806,
    0.,
    -0.07740811,
    0.,
    0.10807149,
    0.,
    -0.15270462,
    0.,
    0.27987896,
    0.,
    -1.01288027,
    -2.01263909,
    -2.25438925,
    -1.93213353,
    -1.73638543,
    -1.85162796,
    -1.91857719,
    -1.7711224,
    -1.65158223,
    -1.69061684,
    -1.71683243,
    -1.61011127,
    -1.51814556,
    -1.52960571,
    -1.53843505,
    -1.44910015,
    -1.37164705,
    -1.36859458,
    -1.36809973,
    -1.28808902,
    -1.21981698,
    -1.20758346,
    -1.20147599,
    -1.12707789,
    -1.06529816,
    -1.04657233,
    -1.03686297,
    -0.96606676,
    -0.90923606,
    -0.8855612,
    -0.87346046,
    -0.80505564,
    -0.75220684,
    -0.72455007,
    -0.71084289,
    -0.64404451,
    -0.59453174,
    -0.56353895,
    -0.5487632,
    -0.48303338,
    -0.43640393,
    -0.40252782,
    -0.38706817,
    -0.32202225,
    -0.27794652,
    -0.24151669,
    -0.22565773,
    -0.16101113,
    -0.11924165,
    -0.08050556,
    -0.03974722,
    0.,
    0.03974722,
    0.08050556,
];

const BEFORE_PROC_DOWN_HC_AD2_2X: &[f32] = &[
    8.62872988e-03,
    0.00000000e+00,
    7.86067752e-03,
    0.00000000e+00,
    7.99297883e-03,
    0.00000000e+00,
    7.14266346e-03,
    0.00000000e+00,
    7.36956604e-03,
    0.00000000e+00,
    6.40970566e-03,
    0.00000000e+00,
    6.76442878e-03,
    0.00000000e+00,
    5.65415670e-03,
    0.00000000e+00,
    6.18755302e-03,
    0.00000000e+00,
    4.86277613e-03,
    0.00000000e+00,
    5.65679840e-03,
    0.00000000e+00,
    4.01100061e-03,
    0.00000000e+00,
    5.20670693e-03,
    0.00000000e+00,
    3.04899490e-03,
    0.00000000e+00,
    4.91136176e-03,
    0.00000000e+00,
    1.86265789e-03,
    0.00000000e+00,
    4.95416329e-03,
    0.00000000e+00,
    1.41293771e-04,
    0.00000000e+00,
    5.89766056e-03,
    0.00000000e+00,
    -3.22668203e-03,
    0.00000000e+00,
    1.02211261e-02,
    0.00000000e+00,
    -1.48777121e-02,
    0.00000000e+00,
    4.23914447e-02,
    0.00000000e+00,
    -2.44333227e-01,
    -8.36486080e-01,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -1.00000000e+00,
    -9.97714829e-01,
    -9.68873951e-01,
    -9.20288009e-01,
    -8.89419239e-01,
    -8.54692431e-01,
    -8.10240979e-01,
    -7.60604185e-01,
    -7.29199936e-01,
    -6.93145825e-01,
    -6.49806381e-01,
    -6.00705066e-01,
    -5.68944631e-01,
    -5.31778511e-01,
    -4.89400173e-01,
    -4.40655044e-01,
    -4.08666639e-01,
    -3.70539413e-01,
    -3.29012313e-01,
    -2.80495154e-01,
    -2.48373645e-01,
    -2.09395182e-01,
    -1.68636837e-01,
    -1.20252782e-01,
    -7.98314788e-02,
    -4.00842606e-02,
    -2.72774353e-18,
    4.00842606e-02,
];

#[test]
fn test_2x_hc_ad2() {
    let mut os = Oversample::new(oversampler::OversampleFactor::TwoTimes, 50);
    os.initialize_oversample_stages();
    let mut ad = ADAASecond::new(NLProc::HardClip);

    let mut input = INPUT_LINSPACE.into_iter().collect::<Vec<f32>>();
    let mut output = vec![0.0_f32; 100];
    let mut result = vec![0.0_f32; 50];

    os.process_up(&mut input, &mut output);

    output
        .iter()
        .zip(OS_2X_PROC_UP.iter())
        .for_each(|(a, b)| assert!((a - b).abs() < ERR_TOL));

    output.iter_mut().for_each(|v| *v = ad.next_adaa(v));

    dbg!(&output);
    // output
    // .iter()
    // .zip(BEFORE_PROC_DOWN_HC_AD2_2X.iter())
    // .for_each(|(a, b)| assert!((a - b).abs() < ERR_TOL));

    os.process_down(&mut output, &mut result);

    let expected_result = [
        -5.58412345e-05,
        7.70732516e-06,
        -5.99214077e-05,
        1.67768959e-05,
        -6.55767677e-05,
        2.76786214e-05,
        -7.33770495e-05,
        4.11096044e-05,
        -8.41839524e-05,
        5.81473256e-05,
        -9.93624762e-05,
        8.05470622e-05,
        -1.21209088e-04,
        1.11374597e-04,
        -1.53915563e-04,
        1.56528103e-04,
        -2.06077482e-04,
        2.29095606e-04,
        -2.98778045e-04,
        3.65688710e-04,
        -5.02676994e-04,
        7.26587196e-04,
        -1.30302756e-03,
        4.53351428e-03,
        8.76375512e-03,
        5.89861662e-03,
        8.16982319e-03,
        5.60557411e-03,
        7.85863041e-03,
        5.19949457e-03,
        7.61294986e-03,
        4.74483220e-03,
        7.41011010e-03,
        4.24770058e-03,
        7.25272601e-03,
        3.69938392e-03,
        7.15506656e-03,
        2.87800808e-03,
        6.64102876e-03,
        1.64968776e-03,
        6.25814190e-03,
        2.34174889e-04,
        6.12805309e-03,
        -1.53274522e-03,
        6.49777467e-03,
        -3.99600615e-03,
        7.48482110e-03,
        -7.56707810e-01,
        -1.04480765e+00,
        -9.87114966e-01,
    ];

    check_results(&result, &expected_result)
}

#[test]
fn real_small_ad2_hc_test() {
    let mut os = Oversample::new(oversampler::OversampleFactor::TwoTimes, 5);
    os.initialize_oversample_stages();
    let mut ad = ADAASecond::new(NLProc::HardClip);

    let mut input = vec![-2., 1., 0., 1., 2.];
    let mut output = vec![0.0_f32; 10];
    // let result = vec![0.0_f32; 5];

    let expected_result = [
        0.00862873,
        0.00862873,
        0.00389146,
        -0.00473727,
        0.00024908,
        0.00498635,
        -0.00458776,
        -0.00957411,
        -0.0081157,
        0.00145841,
    ];

    os.process_up(&mut input, &mut output);

    output.iter_mut().for_each(|v| *v = ad.next_adaa(v));

    dbg!(&output);

    check_results(&output, &expected_result);
}
