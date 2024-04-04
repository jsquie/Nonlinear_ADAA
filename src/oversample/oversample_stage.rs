use ndarray::Array1;

use crate::circular_buffer::CircularBuffer;
use crate::oversample::*;
use num_traits::Float;

use crate::oversample::os_filter_constants::*;

#[allow(dead_code)]
#[derive(Debug)]
pub struct OversampleStage<T>
where
    T: Float + Copy + 'static,
{
    filter_buff: CircularBuffer<T>,
    delay_buff: CircularBuffer<T>,
    pub data: Vec<T>,
    size: usize,
}

#[allow(dead_code)]
impl<T> OversampleStage<T>
where
    T: Float + Copy + From<f32>,
{
    pub fn new(target_size: usize, role: SampleRole) -> Self {
        OversampleStage {
            filter_buff: CircularBuffer::new(NUM_OS_FILTER_TAPS),
            delay_buff: match role {
                SampleRole::UpSample => CircularBuffer::new(UP_DELAY),
                SampleRole::DownSample => CircularBuffer::new(DOWN_DELAY),
            },
            size: target_size,
            data: vec![0.0_f32.into(); target_size],
        }
    }

    pub fn reset(&mut self) {
        self.filter_buff.reset();
        self.delay_buff.reset();
        self.data.iter_mut().for_each(|x| *x = 0.0_32.into());
    }

    pub fn process_up(&mut self, input: &[T], kernel: &Array1<T>) {
        let mut output = self.data.iter_mut();

        input.iter().for_each(|x| {
            match output.next() {
                Some(out) => *out = self.filter_buff.convolve(*x, kernel) * 2.0_f32.into(),
                None => (),
            }
            match output.next() {
                Some(out) => {
                    *out = self
                        .delay_buff
                        .delay(*x * 2.0_f32.into() * FOLD_SCALE_32.into())
                }
                None => (),
            }
        });
    }

    pub fn process_down(&mut self, input: &[T], kernel: &Array1<T>) {
        let output = self.data.iter_mut();
        let mut input_itr = input.into_iter();

        for output_sample in output {
            let even_idx = match input_itr.next() {
                Some(input_sample) => self.filter_buff.convolve(*input_sample, kernel),
                None => 0.0_f32.into(),
            };

            let odd_idx = match input_itr.next() {
                Some(input_sample) => self.delay_buff.delay(*input_sample) * FOLD_SCALE_32.into(),
                None => 0.0_f32.into(),
            };

            *output_sample = even_idx + odd_idx;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_create_os_stage() {
        let _buf: &mut [f32] = &mut [0.0; 8];
        let os_stage = OversampleStage::<f32>::new(8, SampleRole::UpSample);
        assert_eq!(os_stage.data, &[0.0_f32; 8]);
        assert_eq!(os_stage.size, 8);

        let _buf_64: &mut [f64] = &mut [0.0; 8];
        let os_stage_64 = OversampleStage::<f64>::new(8, SampleRole::UpSample);

        assert_eq!(os_stage_64.data, &[0.0_f64; 8]);
        assert_eq!(os_stage_64.size, 8);
    }

    #[test]
    fn test_os_stage_up() {
        let _buf: &mut [f64] = &mut [0.0; 8];
        let mut os_stage = OversampleStage::new(8, SampleRole::UpSample);
        let kern = get_kern();

        let signal: &[f64] = &[1., 0., 0., 0.];

        os_stage.process_up(signal, &kern);

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

        assert_eq!(expected_result, os_stage.data);
    }

    #[test]
    fn test_os_stage_down() {
        let _buf: &mut [f64] = &mut [0.0; 8];
        let mut os_stage = OversampleStage::new(8, SampleRole::DownSample);

        let kern = get_kern();

        let signal_vec: Vec<f64> = vec![vec![1.], vec![0.; 15]].into_iter().flatten().collect();

        let signal: &[f64] = signal_vec.as_slice();

        os_stage.process_down(signal, &kern);

        let expected_result: &[f64] = &[
            -0.0064715474097890545,
            0.006788724784527351,
            -0.007134125572070907,
            0.007511871271766723,
            -0.007926929217098087,
            0.00838534118242672,
            -0.00889453036904902,
            0.009463720022395613,
        ];

        assert_eq!(expected_result, os_stage.data);
    }

    #[test]
    fn test_multi_stage_up_small() {
        let _buf_0: &mut [f64] = &mut [0.0; 8];
        let _buf_1: &mut [f64] = &mut [0.0; 16];

        let mut os_stage_0 = OversampleStage::new(8, SampleRole::UpSample);
        let mut os_stage_1 = OversampleStage::new(16, SampleRole::UpSample);

        let kern = get_kern();

        let signal: &[f64] = &[1., 0., 0., 0.];

        os_stage_0.process_up(signal, &kern);
        os_stage_1.process_up(&os_stage_0.data, &kern);

        let expected_result: &[f64] = &[
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

        assert_eq!(
            expected_result
                .into_iter()
                .map(|x| *x as f32)
                .collect::<Vec<_>>(),
            os_stage_1
                .data
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        );
    }

    #[test]
    fn test_os_multi_stage_down() {
        let _buf_0: &mut [f64] = &mut [0.0; 16];
        let _buf_1: &mut [f64] = &mut [0.0; 8];

        let mut os_stage_0 = OversampleStage::new(16, SampleRole::DownSample);
        let mut os_stage_1 = OversampleStage::new(8, SampleRole::DownSample);

        let kern = get_kern();

        let signal_vec: Vec<f64> = vec![vec![1.], vec![0.; 31]].into_iter().flatten().collect();
        let signal: &[f64] = signal_vec.as_slice();

        os_stage_0.process_down(signal, &kern);
        os_stage_1.process_down(&os_stage_0.data, &kern);

        let expected_result: &[f64] = &[
            4.188092587714742e-05,
            2.2352775719665985e-06,
            4.903671502248951e-05,
            6.029950736406798e-06,
            5.926989699611419e-05,
            1.3084963435659304e-05,
            7.56903059770508e-05,
            2.835997202417606e-05,
        ];

        let vec32_er = slice_to_f32_vec(expected_result);
        let display_osr = slice_to_f32_vec(&os_stage_1.data);

        assert_eq!(vec32_er, display_osr);
    }

    fn slice_to_f32_vec(arr: &[f64]) -> Vec<f32> {
        arr.into_iter().map(|x| *x as f32).collect::<Vec<f32>>()
    }

    const RAND_TEST_DATA: &[f64] = &[
        2.542913180922093,
        0.5862253930784518,
        -0.3781017981702173,
        0.9719707041773196,
        -0.723438732761163,
        0.46736570485331075,
        1.1230884239093122,
        -1.5254702810737608,
        -1.0369418785020552,
        -0.7551785221976162,
        0.355800385238018,
        -0.5496093998991464,
        1.8590280730593869,
        2.087612808960415,
        -0.46388441828587534,
        -0.3128312459772416,
        1.0149635867592255,
        0.3506127205242301,
        -0.029696502629996187,
        -0.2829972749482435,
        1.5890446250525057,
        -0.008351621752887797,
        -1.1129456636678388,
        -1.5361537403069268,
        -1.041077829067189,
        0.728072050969641,
        -0.8438129596070822,
        0.4402591181465738,
        -0.42971887492521316,
        0.15334135252187214,
        -0.7635461199104077,
        1.762123972569591,
        0.3301145895344221,
        0.2014572416094927,
        0.7764788596949593,
        0.39160350248318776,
        -0.17756425187058367,
        0.27433281620925154,
        0.4080337326257071,
        -1.6414140095086096,
        -1.336345359235497,
        -1.1008971772091236,
        -1.3157063518867227,
        1.0175764232139108,
        0.17440562421604466,
        -1.0313598972790907,
        0.23651759499372582,
        -1.3146833809038507,
        -1.124634252813258,
        1.1031223369325283,
        -0.0004804077318948717,
        0.7880973652232677,
        0.5194246080123744,
        0.34318373072487535,
        1.4884272487515111,
        0.7985053660609663,
        -0.01826882502069841,
        -0.2802137889249463,
        -1.24550136558111,
        0.6171569477185286,
        -0.5509759086738899,
        0.4617242023903233,
        -0.5468219478819403,
        -0.3319779579370651,
    ];

    #[test]
    fn test_big_rand_os_stage_up() {
        let mut os_stage_0 = OversampleStage::new(RAND_TEST_DATA.len() * 2, SampleRole::UpSample);

        let kern = get_kern();

        os_stage_0.process_up(RAND_TEST_DATA, &kern);

        let expected_rand_conv_upsample: &[f64] = &[
            -0.03291316641862963,
            0.0,
            0.026938704624195346,
            0.0,
            -0.023429670767352893,
            0.0,
            0.012125694722570428,
            0.0,
            -0.0035524154555044056,
            0.0,
            -0.0020679828803609156,
            0.0,
            -0.0126760273113218,
            0.0,
            0.033407660951992726,
            0.0,
            -0.022091759539897612,
            0.0,
            0.033586242017822864,
            0.0,
            -0.040665536296736624,
            0.0,
            0.05086909013493954,
            0.0,
            -0.07888566569840653,
            0.0,
            0.05769957933448332,
            0.0,
            -0.05723486800651065,
            0.0,
            0.06779258094595116,
            0.0,
            -0.08940279608016921,
            0.0,
            0.09664030957887666,
            0.0,
            -0.11209120170098845,
            0.0,
            0.13892448198370205,
            0.0,
            -0.19713629918927675,
            0.0,
            0.269287948836763,
            0.0,
            -0.4349225617353005,
            0.0,
            1.4605791421491887,
            2.558983238061923,
            2.127040522222565,
            0.5899300714899162,
            -0.5491294710800522,
            -0.38049122992386636,
            0.558354512701653,
            0.978113117874946,
            0.28668067386539803,
            -0.7280105372017781,
            -0.7410288237351733,
            0.47031924398258074,
            1.5530649381965325,
            1.1301858330071566,
            -0.4242190701538356,
            -1.5351105608779967,
            -1.5003877025941268,
            -1.043494880532651,
            -0.9257284262548935,
            -0.7599509077015868,
            -0.11302200855181459,
            0.3580488821839795,
            -0.057197128801057595,
            -0.5530826818527852,
            0.17752849116517916,
            1.8707762867155764,
            2.814809189882358,
            2.100805574398703,
            0.5895768984965585,
            -0.4668159572641065,
            -0.7090478440118182,
            -0.3148081974656741,
            0.4609644677463938,
            1.021377695961394,
            0.90948944535792,
            0.3528284337837495,
            0.04018914218656989,
            -0.029884171048130743,
            -0.22829124890194985,
            -0.2847856892806532,
            0.48359143032798785,
            1.5990866658559098,
            1.5078827247592457,
            -0.008404400211777413,
            -1.2189244596622777,
            -1.1199789751244977,
            -0.8693087004863899,
            -1.5458615347245659,
            -2.052913446363682,
            -1.0476569684281536,
            0.5252895687296472,
            0.7326731358783884,
            -0.38138295414562023,
            -0.8491454745265645,
            -0.08266417501565877,
            0.44304135595084726,
            -0.059897402618353884,
            -0.43243450317627585,
            -0.014983361829685344,
            0.1543104002720717,
            -0.5248832986461686,
            -0.7683713848340834,
            0.37591754989473464,
            1.7732597962929908,
            1.657040845908973,
            0.332200763909666,
            -0.3387729622883874,
            0.2027303599401481,
            0.8497385918015886,
            0.7813858536642319,
            0.4839787629153856,
            0.3940782588800147,
            0.2176549623263,
            -0.17868637735050932,
            -0.27121087829296064,
            0.27606647509500815,
            0.7479634862076271,
            0.4106123206197713,
            -0.6545146416918808,
            -1.6517870010526452,
            -1.846242820375438,
            -1.3447904553727055,
        ];

        for (a, b) in expected_rand_conv_upsample
            .iter()
            .zip(os_stage_0.data.iter())
        {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_big_rand_os_stage_down() {
        let mut os_stage_0 = OversampleStage::new(32, SampleRole::DownSample);

        let kern = get_kern();

        os_stage_0.process_down(RAND_TEST_DATA, &kern);

        let expected_rand_downsample: &[f64] = &[
            -0.016456583209314816,
            0.019710041448812148,
            -0.016026522942743406,
            0.009620115741640595,
            -0.0035016853161247033,
            0.0015316562398641398,
            -0.013835071643340057,
            0.01775431469808623,
            -0.025508641012424862,
            0.027370052308072053,
            -0.03955923803466271,
            0.04945258766425772,
            -0.046157528962171124,
            0.05526717501871183,
            -0.05706606682168856,
            0.06736178499063673,
            -0.07634052795886197,
            0.0800906718287717,
            -0.09028829992242991,
            0.10361188136060669,
            -0.11956169978277063,
            0.17189304969551691,
            -0.27772729830842224,
            0.8146007117140612,
            1.0844720828663474,
            -0.20338240527631907,
            0.5306420434242592,
            -0.6295660034211816,
            -0.9751636752702594,
            0.740482648639267,
            1.1991247554776259,
            0.010755973780039866,
        ];

        for (a, b) in expected_rand_downsample.iter().zip(os_stage_0.data.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
