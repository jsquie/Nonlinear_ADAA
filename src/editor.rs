use crate::AtomicF32;
use nih_plug::prelude::{util, Editor};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::NonlinearAdaaParams;

const STYLE: &str = r#"
    .foo {
        width: 300px; 
    }
    
"#;

#[derive(Lens)]
struct Data {
    params: Arc<NonlinearAdaaParams>,
    input_peak_meters: [Arc<AtomicF32>; 2],
    output_peak_meters: [Arc<AtomicF32>; 2],
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (500, 600))
}

pub(crate) fn create(
    params: Arc<NonlinearAdaaParams>,
    input_meters: [Arc<AtomicF32>; 2],
    output_meters: [Arc<AtomicF32>; 2],
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);
        assets::register_noto_sans_bold(cx);

        let _ = cx.add_stylesheet(STYLE);

        Data {
            params: params.clone(),
            input_peak_meters: input_meters.clone(),
            output_peak_meters: output_meters.clone(),
        }
        .build(cx);

        VStack::new(cx, |cx| {
            HStack::new(cx, |cx| {
                VStack::new(cx, |_| {});
                VStack::new(cx, |cx| {
                    Label::new(cx, "Nonlinear ADAA")
                        .font_size(30.0)
                        .height(Pixels(50.0))
                        .width(Pixels(300.0))
                        .text_align(TextAlign::Center);
                })
                .class("foo");
                VStack::new(cx, |_| {});
            })
            .height(Pixels(100.0));

            HStack::new(cx, |cx| {
                VStack::new(cx, |cx| {
                    Label::new(cx, "Bypass");
                    ParamButton::new(cx, Data::params, |params| &params.bypass);

                    Label::new(cx, "Gain").top(Pixels(10.0));
                    ParamSlider::new(cx, Data::params, |params| &params.gain);

                    Label::new(cx, "Output");
                    ParamSlider::new(cx, Data::params, |params| &params.output);

                    Label::new(cx, "Prefilter Cutoff Frequency");
                    ParamSlider::new(cx, Data::params, |params| &params.pre_filter_cutoff);

                    Label::new(cx, "Clip Style");
                    ParamSlider::new(cx, Data::params, |params| &params.nl_proc_type);

                    Label::new(cx, "ADAA order");
                    ParamSlider::new(cx, Data::params, |params| &params.nl_proc_order);

                    Label::new(cx, "Oversampling");
                    ParamSlider::new(cx, Data::params, |params| &params.os_level);

                    Label::new(cx, "Mix");
                    ParamSlider::new(cx, Data::params, |params| &params.dry_wet);
                })
                .border_width(Pixels(5.0))
                .border_color(Color::black())
                .left(Pixels(5.0));

                VStack::new(cx, |cx| {
                    Label::new(cx, "Input Level").top(Pixels(10.0));
                    PeakMeter::new(
                        cx,
                        Data::input_peak_meters.map(|peak_meter| {
                            util::gain_to_db(peak_meter[0].load(Ordering::Relaxed))
                        }),
                        Some(Duration::from_millis(600)),
                    );
                    PeakMeter::new(
                        cx,
                        Data::input_peak_meters.map(|peak_meter| {
                            util::gain_to_db(peak_meter[1].load(Ordering::Relaxed))
                        }),
                        Some(Duration::from_millis(600)),
                    )
                    .bottom(Pixels(5.0));

                    Label::new(cx, "Output Level");
                    PeakMeter::new(
                        cx,
                        Data::output_peak_meters.map(|peak_meter| {
                            util::gain_to_db(peak_meter[0].load(Ordering::Relaxed))
                        }),
                        Some(Duration::from_millis(600)),
                    );
                    PeakMeter::new(
                        cx,
                        Data::output_peak_meters.map(|peak_meter| {
                            util::gain_to_db(peak_meter[1].load(Ordering::Relaxed))
                        }),
                        Some(Duration::from_millis(600)),
                    );
                })
                .right(Pixels(5.0))
                .border_width(Pixels(5.0))
                .border_color(Color::blue());
            });
        });
    })
}
