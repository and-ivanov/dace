from dace.config import Config

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Pango

_ALLOWED_TYPES = {'int': int, 'str': str, 'float': float}


class DIODEConfig:
    """ This class holds all configuration options for DIODE. The "Preferences"
        dialog is autogenerated from the description of options in dace.Config.
        Options are described as a list of dictionaries. Each option is one
        dictionary with the entry's name, category, details, defualt and type. 
        
        @see dace.config.Config
    """

    def __init__(self):
        self.window = None

    def __getitem__(self, *key):
        return Config.get(*key)

    def __setitem__(self, *key, value=None):
        Config.set(*key, value=value)

    def textfield_callback(self, widget, cpath, meta):
        value = widget.get_text()
        casted = _ALLOWED_TYPES[meta['type']](value)
        Config.set(*cpath, value=casted)

    def switch_callback(self, widget, data, cpath):
        value = widget.get_active()
        Config.set(*cpath, value=value)

    def fontbutton_callback(self, widget, cpath):
        value = widget.get_font_name()
        Config.set(*cpath, value=value)

    def win_close_callback(self, widget, *data):
        Config.save()

    def render_config_element(self, cval, cpath, grid, i, meta):
        grid.insert_row(i)
        label = Gtk.Label()
        # If setting was modified from default, mark label as bold
        if cval != Config.get_default(*cpath):
            label.set_markup('<b>' + meta['title'] + '</b>')
        else:
            label.set_label(meta['title'])

        entry = None
        if (meta['type'] == "str" or meta['type'] == "int"
                or meta['type'] == "float"):

            entry = Gtk.Entry()
            entry.set_text(str(cval))
            entry.connect("changed", self.textfield_callback, cpath, meta)
        elif meta['type'] == "bool":
            entry = Gtk.Switch()
            entry.set_active(cval)
            entry.connect("state-set", self.switch_callback, cpath)
        elif meta['type'] == "font":
            entry = Gtk.FontButton()
            entry.set_use_font(True)
            entry.set_font_name(str(cval))
            entry.connect("font-set", self.fontbutton_callback, cpath)
        else:
            raise ValueError("Unimplemented CV type: " + meta['type'])
        label.set_tooltip_text(meta['description'])
        entry.set_tooltip_text(meta['description'])
        grid.attach(label, 0, i, 1, 1)
        grid.attach(entry, 1, i, 1, 1)

    def render_config_subtree(self, cv, config_path, grid):
        # Add notebook to grid and render each child within

        columized = False
        notebook = Gtk.Notebook()
        grid.add(notebook)
        grid.set_hexpand(True)
        for i, (cname, cval) in enumerate(sorted(cv.items())):
            # Create current config "path"
            cpath = tuple(list(config_path) + [cname])
            meta = Config.get_metadata(*cpath)
            if meta['type'] == 'dict':
                gtklabel = Gtk.Label()
                gtklabel.set_label(meta['title'])
                ngrid = Gtk.Grid()
                notebook.append_page(ngrid, gtklabel)
                self.render_config_subtree(cval, cpath, ngrid)
                continue

            if columized == False:
                grid.insert_column(0)
                grid.insert_column(1)
                columized = True
            self.render_config_element(cval, cpath, grid, i, meta)

    def render_config_dialog(self):
        # Load metadata for configuration
        Config.load_schema()

        self.window = Gtk.Window()
        notebook = Gtk.Notebook()
        notebook.set_scrollable(True)
        self.window.add(notebook)

        # General (top-level) settings
        gtklabel = Gtk.Label()
        gtklabel.set_label('General')
        general_grid = Gtk.Grid()
        general_grid.set_hexpand(True)
        notebook.append_page(general_grid, gtklabel)
        columized = False

        for i, (cname, cval) in enumerate(sorted(Config.get().items())):
            meta = Config.get_metadata(cname)
            if meta['type'] == 'dict':
                gtklabel = Gtk.Label()
                gtklabel.set_label(meta['title'])
                grid = Gtk.Grid()
                grid.set_hexpand(True)
                notebook.append_page(grid, gtklabel)
                self.render_config_subtree(cval, (cname, ), grid)
                continue

            if columized == False:
                general_grid.insert_column(0)
                general_grid.insert_column(1)
                columized = True
            self.render_config_element(cval, (cname, ), general_grid, i, meta)

        self.window.show_all()
        self.window.connect("delete-event", self.win_close_callback, None)
