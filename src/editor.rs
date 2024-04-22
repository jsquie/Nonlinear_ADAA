use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::Arc;

use crate::NonlinearAdaaParams;

#[derive(Lens)]
struct Data {
    params: Arc<NonlinearAdaaParams>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (500, 500))
}

pub(crate) fn create(
    params: Arc<NonlinearAdaaParams>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);

        Data {
            params: params.clone(),
        }
        .build(cx);

        VStack::new(cx, |cx| {
            Label::new(cx, "Nonlinear ADAA")
                .font_size(30.0)
                .height(Pixels(50.0));

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
        })
        .top(Pixels(10.0));
    })
}
