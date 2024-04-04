#[cfg(test)]
mod tests {

    /*
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
    fn test_up_sample_2x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::TwoTimes, 8);
        let sig = range_to_float_vec(0, 8);
        let up_sampled_2x = os.process_up(&sig);
        let expected_result = iter_collect_32f(vec![
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
            -0.03901680260122647,
            0.0,
            -0.03671013252170096,
            0.0,
            -0.05219252318027351,
            0.0,
        ]);

        assert_eq!(iter_collect_32f(up_sampled_2x), expected_result);
    }

    #[test]
    fn test_up_sample_4x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::FourTimes, 8);
        let sig = range_to_float_vec(0, 8);
        let up_sampled_4x = os.process_up(&sig);
        let expected_result = iter_collect_32f(vec![
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
        ]);

        assert_eq!(iter_collect_32f(up_sampled_4x), expected_result);
    }

    #[test]
    fn test_up_sample_8x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 8);
        let sig = range_to_float_vec(0, 8);
        let up_sampled_8x = os.process_up(&sig);
        let expected_result = iter_collect_32f(vec![
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
        ]);

        assert_eq!(iter_collect_32f(up_sampled_8x), expected_result);
    }

    #[test]
    fn test_up_sample_16x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 8);
        let sig = range_to_float_vec(0, 8);
        let up_sampled_16x = os.process_up(&sig);
        let expected_result = iter_collect_32f(vec![
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
            2.806419123723386e-08,
            0.0,
            -2.943964689522919e-08,
            0.0,
            1.4978479089021315e-09,
            0.0,
            -1.6930929654312068e-09,
            0.0,
            3.419601538621763e-09,
            0.0,
            -3.762540309236139e-09,
            0.0,
            2.469379700187792e-09,
            0.0,
            -2.8567917912405475e-09,
            0.0,
            6.287296607853106e-08,
            0.0,
            -6.63630120817841e-08,
            0.0,
            7.605130721855296e-09,
            0.0,
            -8.893842311506477e-09,
            0.0,
            1.5956566524510785e-08,
            0.0,
            -1.8236965037909117e-08,
            0.0,
            1.487002006005615e-08,
            0.0,
            -1.8280420790377385e-08,
            0.0,
            1.5011240354650044e-07,
            0.0,
            -1.6283158650824303e-07,
            0.0,
            4.497693339235821e-08,
            0.0,
            -6.091777977154983e-08,
            0.0,
            1.0377846923823098e-07,
            0.0,
            -1.5897421719027295e-07,
            0.0,
            2.8652953108822606e-07,
            0.0,
            -1.137182264312365e-06,
            -2.1819776940451505e-06,
            -1.6167289740242459e-06,
            0.0,
            1.6356326346708743e-06,
            2.288918725750784e-06,
            1.2821127025461411e-06,
            0.0,
            -4.680756273914324e-07,
            -1.1645697175696516e-07,
            2.54072188353592e-07,
            0.0,
            -3.09315245826344e-07,
            1.316371832448989e-07,
            7.233087726310213e-07,
            0.0,
            -2.4609632861026158e-06,
            -4.629827802396532e-06,
            -3.4942576267286723e-06,
            0.0,
            3.551780532681899e-06,
            4.870373194231498e-06,
            2.74874640608505e-06,
            0.0,
            -1.0769553488161305e-06,
            -4.249070559606927e-07,
            3.8779805338618434e-07,
            0.0,
            -4.881732642216914e-07,
            4.853885871464764e-07,
            1.61469099249281e-06,
            0.0,
            -5.237644359274428e-06,
            -9.890984845490228e-06,
            -7.680345973384166e-06,
            0.0,
            7.838991986245219e-06,
            1.0434784297040746e-05,
            5.777404317562723e-06,
            0.0,
            -2.269785202435481e-06,
            -1.2139031910305736e-06,
            5.271277284022413e-07,
            0.0,
            -5.983763153388258e-07,
            1.4054465497404469e-06,
            2.124649567488963e-06,
            0.0,
            -9.916277314528848e-06,
            -1.6139915718674515e-05,
            -1.3065527698190472e-05,
            0.0,
            1.3405460142692664e-05,
            1.7138912613299124e-05,
            1.090298524580925e-05,
            0.0,
            -3.2756928970308225e-06,
            -2.9764530778889845e-06,
            2.9724100523248675e-07,
            0.0,
            -3.278228812151181e-07,
            3.5355704790248353e-06,
            3.7689179615161576e-06,
            0.0,
            -1.714365613959223e-05,
            -2.701112037959445e-05,
        ]);

        assert_eq!(iter_collect_32f(up_sampled_16x), expected_result);
    }

    #[test]
    fn test_down_sample_2x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::TwoTimes, 8);
        let sig = range_to_float_vec(0, 16);
        let down_sampled_2x = os.process_down(&sig);
        let expected_result = iter_collect_32f(vec![
            0.0,
            -0.012943094819578109,
            -0.012308740070101515,
            -0.025942636464766737,
            -0.024552790315898507,
            -0.03901680260122647,
            -0.03671013252170096,
            -0.05219252318027351,
        ]);

        assert_eq!(iter_collect_32f(down_sampled_2x), expected_result);
    }

    #[test]
    fn test_down_sample_4x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::FourTimes, 8);
        let sig = range_to_float_vec(0, 32);
        let down_sampled_4x = os.process_down(&sig);
        let expected_result = iter_collect_32f(vec![
            0.0,
            7.965659491843221e-05,
            7.533389779174369e-05,
            0.00015870132418106852,
            0.0001489576196064275,
            0.0002363480222189285,
            0.00021928636423278896,
            0.00031025949915092433,
        ]);

        assert_eq!(iter_collect_32f(down_sampled_4x), expected_result);
    }

    #[test]
    fn test_down_sample_8x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 8);
        let sig = range_to_float_vec(0, 64);
        let down_sampled_8x = os.process_down(&sig);
        let expected_result = iter_collect_32f(vec![
            0.0,
            -4.875268911234722e-07,
            -4.5256519827845846e-07,
            -9.453313050662675e-07,
            -8.289969026739245e-07,
            -1.1198132979211948e-06,
            3.506333469088274e-05,
            0.00015926393646145928,
        ]);

        assert_eq!(iter_collect_32f(down_sampled_8x), expected_result);
    }

    #[test]
    fn test_down_sample_16x_small() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 8);
        let sig = range_to_float_vec(0, 128);
        let down_sampled_16x = os.process_down(&sig);
        let expected_result = iter_collect_32f(vec![
            0.0,
            -4.875268911234724e-07,
            -4.5256519827845804e-07,
            -9.45331305066268e-07,
            -8.289969026739222e-07,
            -1.1198132979211957e-06,
            3.506333469088274e-05,
            0.00015926393646145936,
        ]);

        assert_eq!(iter_collect_32f(down_sampled_16x), expected_result);
    }
    */
    /*
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
        let sig_to_down_sample = range_to_float_vec(0, 2 << 7).as_mut_slice();

        os_stage_0.process_down(&sig_to_down_sample, &get_kern());
        os_stage_1.process_down(&os_stage_0.get_processed_data(), &get_kern());
        os_stage_2.process_down(&os_stage_1.get_processed_data(), &get_kern());

        let result = iter_collect_32f(os_stage_2.get_processed_data().to_vec()).as_slice();
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
        ]
        .as_slice();

        assert_eq!(result, expected_result);
    }
    */
}
