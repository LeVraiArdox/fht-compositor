use glib::object::{Cast, ObjectExt};
use gtk::prelude::{ButtonExt, ListBoxRowExt, ToggleButtonExt, WidgetExt};
use gtk::subclass::box_::BoxImpl;
use gtk::subclass::prelude::*;
use gtk::{gio, glib};
use output_button::OutputButton;

mod output_button;
mod output_grid;
mod window_row;
mod workspace_row;

use output_grid::OutputGrid;
use window_row::WindowRow;
use workspace_row::WorkspaceRow;

use crate::output_object::OutputObject;
use crate::utils::forall_siblings;
use crate::window_object::WindowObject;
use crate::ScreencastSource;

mod imp {
    use std::cell::RefCell;
    use std::sync::OnceLock;

    use glib::subclass::Signal;
    use glib::types::StaticType;

    use super::*;

    #[derive(Default, Debug, gtk::CompositeTemplate)]
    #[template(resource = "/fht/desktop/SharePicker/ui/selection-widget.ui")]
    pub struct SelectionWidget {
        #[template_child]
        pub source_type: TemplateChild<adw::ViewStack>,
        #[template_child]
        pub window_list: TemplateChild<gtk::ListBox>,
        #[template_child]
        pub workspace_list: TemplateChild<gtk::ListBox>,
        #[template_child]
        pub output_grid: TemplateChild<OutputGrid>,

        // Data gathered for selection
        pub windows: RefCell<Option<gio::ListStore>>,
        pub outputs: RefCell<Option<gio::ListStore>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for SelectionWidget {
        const NAME: &'static str = "SelectionWidget";
        type Type = super::SelectionWidget;
        type ParentType = gtk::Box;

        fn class_init(klass: &mut Self::Class) {
            klass.bind_template();
        }

        fn instance_init(obj: &glib::subclass::InitializingObject<Self>) {
            obj.init_template();
        }
    }

    impl ObjectImpl for SelectionWidget {
        fn constructed(&self) {
            self.parent_constructed();

            let obj = self.obj();
            obj.setup_values();
        }

        fn signals() -> &'static [Signal] {
            static SIGNALS: OnceLock<Vec<Signal>> = OnceLock::new();
            SIGNALS.get_or_init(|| {
                vec![Signal::builder("selection-changed")
                    .param_types([Option::<String>::static_type()])
                    .build()]
            })
        }
    }

    impl SelectionWidget {
        pub fn set_selection(&self, selection: Option<ScreencastSource>) {
            if let Some(selection) = selection {
                let data = serde_json::ser::to_string(&selection).unwrap();
                self.obj()
                    .emit_by_name::<()>("selection-changed", &[&Some(data)])
            } else {
                self.obj()
                    .emit_by_name::<()>("selection-changed", &[&Option::<String>::None])
            }
        }
    }

    impl WidgetImpl for SelectionWidget {}
    impl BoxImpl for SelectionWidget {}
}

glib::wrapper! {
    pub struct SelectionWidget(ObjectSubclass<imp::SelectionWidget>)
        @extends gtk::Widget, gtk::Box,
        @implements gio::ActionMap, gio::ActionGroup;
}

impl SelectionWidget {
    fn setup_values(&self) {
        let imp = self.imp();
        // Gather info from compositor.
        // This program shouldn't be used outside of the compositor so its fine if we panic.
        let (windows, outputs) = crate::get_compositor_data().unwrap();

        let model = gio::ListStore::new::<WindowObject>();
        model.extend_from_slice(&windows);
        imp.windows.replace(Some(model));

        let model = gio::ListStore::new::<OutputObject>();
        model.extend_from_slice(&outputs);
        imp.outputs.replace(Some(model));

        imp.window_list
            .bind_model(Some(imp.windows.borrow().as_ref().unwrap()), |obj| {
                let window_obj = obj.downcast_ref::<WindowObject>().unwrap();
                let window_row = WindowRow::new();
                window_row.bind_window(window_obj);
                window_row.upcast()
            });
        imp.window_list.connect_row_selected(glib::clone!(
            #[weak]
            imp,
            move |_, row| {
                let selection = row.map(|row| {
                    let row = row.downcast_ref::<WindowRow>().unwrap();
                    let id = row.identifier() as usize;
                    ScreencastSource::Window { id }
                });

                imp.set_selection(selection);
            }
        ));

        // The rows in the workspace list are all expander rows, they dont provide much.
        // When expanded, the workspace row shows an index selection list
        imp.workspace_list
            .set_selection_mode(gtk::SelectionMode::None);
        imp.workspace_list.bind_model(
            Some(imp.outputs.borrow().as_ref().unwrap()),
            glib::clone!(
                #[weak]
                imp,
                #[upgrade_or_panic]
                move |obj| {
                    let output_obj = obj.downcast_ref::<OutputObject>().unwrap();

                    let workspace_row = WorkspaceRow::new();
                    workspace_row.set_output(output_obj);

                    workspace_row.connect_workspace_selected(glib::clone!(
                        #[weak]
                        imp,
                        move |row, index| {
                            imp.set_selection(Some(ScreencastSource::Workspace {
                                output: row.output_name(),
                                idx: index,
                            }));
                        }
                    ));

                    workspace_row.upcast()
                }
            ),
        );
        imp.workspace_list.connect_row_activated(glib::clone!(
            #[weak]
            imp,
            move |_, row| {
                imp.set_selection(None);
                let child = row.child().unwrap();
                let workspace_row: &WorkspaceRow = child.downcast_ref().unwrap();

                if workspace_row.expanded() {
                    forall_siblings(row, |sibling| {
                        let row: &gtk::ListBoxRow = sibling.downcast_ref().unwrap();
                        let child = row.child().unwrap();
                        let workspace_row: &WorkspaceRow = child.downcast_ref().unwrap();
                        workspace_row.set_expanded(false);
                        workspace_row.deselect_all_buttons();
                    });
                }
            }
        ));

        for output in &outputs {
            let output_button = OutputButton::new();
            output_button.set_output(output);

            output_button.connect_clicked(glib::clone!(
                #[weak]
                imp,
                move |button| {
                    if button.is_active() {
                        imp.set_selection(Some(ScreencastSource::Output {
                            name: button.output_name(),
                        }));

                        forall_siblings(button, |sibling| {
                            let sibling = sibling.downcast_ref::<gtk::ToggleButton>().unwrap();
                            sibling.set_active(false);
                        });
                    } else {
                        imp.set_selection(None);
                    }
                }
            ));

            // FIXME: I don't really like how much duplication there is within the output grid code
            // But for now I don't really know how todo it better ngl
            imp.output_grid
                .add_output(&output_button.upcast(), output.rect());
        }

        // Whenever there's a page switch, we reset the selection
        imp.source_type.connect_visible_child_notify(glib::clone!(
            #[weak]
            imp,
            move |_| {
                imp.window_list.select_row(Option::<&adw::ActionRow>::None);

                if let Some(first_row) = imp.workspace_list.first_child() {
                    let child = first_row
                        .downcast_ref::<gtk::ListBoxRow>()
                        .unwrap()
                        .child()
                        .unwrap();
                    let workspace_row = child.downcast_ref::<WorkspaceRow>().unwrap();
                    workspace_row.deselect_all_buttons();
                    workspace_row.set_expanded(false);

                    forall_siblings(&first_row, |sibling_row| {
                        let child = sibling_row
                            .downcast_ref::<gtk::ListBoxRow>()
                            .unwrap()
                            .child()
                            .unwrap();
                        let row = child.downcast_ref::<WorkspaceRow>().unwrap();
                        row.deselect_all_buttons();
                        row.set_expanded(false);
                    });

                    imp.output_grid.deselect_all();
                }

                imp.set_selection(None);
            }
        ));

        imp.set_selection(None);
    }
}
