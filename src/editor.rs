use crate::AtomicF32;
use nih_plug::prelude::{util, Editor};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::NonlinearAdaaParams;

// use crate::custom_widgets::knob;

#[derive(Lens)]
struct Data {
    params: Arc<NonlinearAdaaParams>,
    input_peak_meters: [Arc<AtomicF32>; 2],
    output_peak_meters: [Arc<AtomicF32>; 2],
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (430, 600))
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

        // let _ = cx.add_stylesheet(STYLE);

        Data {
            params: params.clone(),
            input_peak_meters: input_meters.clone(),
            output_peak_meters: output_meters.clone(),
        }
        .build(cx);

        VStack::new(cx, |cx| {
            // Knob::new(cx, 0.5, , false);

            HStack::new(cx, |cx| {
                Label::new(cx, "Nonlinear ADAA").font_size(30.0);
            })
            .child_space(Stretch(1.0))
            //.border_width(Pixels(5.0))
            //.border_color(Color::black())
            .height(Pixels(50.0));

            HStack::new(cx, |cx| {
                VStack::new(cx, |cx| {
                    VStack::new(cx, |cx| {
                        ParamButton::new(cx, Data::params, |params| &params.bypass);

                        VStack::new(cx, |cx| {
                            Label::new(cx, "Gain");
                            ParamSlider::new(cx, Data::params, |params| &params.gain);
                        })
                        .row_between(Pixels(2.0));

                        VStack::new(cx, |cx| {
                            Label::new(cx, "Output");
                            ParamSlider::new(cx, Data::params, |params| &params.output);
                        })
                        .row_between(Pixels(2.0));

                        VStack::new(cx, |cx| {
                            Label::new(cx, "Prefilter Cutoff Frequency");
                            ParamSlider::new(cx, Data::params, |params| &params.pre_filter_cutoff);
                        })
                        .row_between(Pixels(2.0));

                        VStack::new(cx, |cx| {
                            Label::new(cx, "Clip Style");
                            ParamSlider::new(cx, Data::params, |params| &params.nl_proc_type);
                        })
                        .row_between(Pixels(2.0));

                        VStack::new(cx, |cx| {
                            Label::new(cx, "ADAA order");
                            ParamSlider::new(cx, Data::params, |params| &params.nl_proc_order);
                        })
                        .row_between(Pixels(2.0));

                        VStack::new(cx, |cx| {
                            Label::new(cx, "Oversampling");
                            ParamSlider::new(cx, Data::params, |params| &params.os_level);
                        })
                        .row_between(Pixels(2.0));

                        VStack::new(cx, |cx| {
                            Label::new(cx, "Mix");
                            ParamSlider::new(cx, Data::params, |params| &params.dry_wet);
                        })
                        .row_between(Pixels(2.0));
                    })
                    .child_space(Percentage(5.0))
                    //.border_width(Pixels(5.0))
                    //.border_color(Color::black())
                    .row_between(Pixels(10.0));
                })
                .child_left(Percentage(5.0))
                .child_bottom(Percentage(10.0));
                //.border_width(Pixels(5.0))
                //.border_color(Color::black());

                VStack::new(cx, |cx| {
                    VStack::new(cx, |cx| {
                        Label::new(cx, "Input Level");
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
                        );
                    })
                    //.border_width(Pixels(5.0))
                    //.border_color(Color::black())
                    // .space(Percentage(10.0))
                    .row_between(Pixels(5.0));

                    VStack::new(cx, |cx| {
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
                    //.border_width(Pixels(5.0))
                    //.border_color(Color::black())
                    .row_between(Pixels(5.0));
                    // .space(Percentage(10.0));
                })
                .child_space(Percentage(1.0))
                .child_bottom(Percentage(65.0))
                //.border_width(Pixels(5.0))
                //.border_color(Color::black())
                .row_between(Pixels(10.0));
            })
            //.border_width(Pixels(5.0))
            //.border_color(Color::black())
            .col_between(Pixels(20.0));
        })
        //.border_width(Pixels(5.0))
        //.border_color(Color::black())
        .row_between(Pixels(20.0));
    })
}
