"""Main UI class for Labquake Explorer"""
import sys
import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog, simpledialog, messagebox
from pathlib import Path
from typing import Optional, List, Dict, Any

from labquake_explorer._version import __version__
from labquake_explorer.data.channels import is_channel_array
from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.utils.config import LabquakeExplorerConfig
from labquake_explorer.ui.context import (
    TreeContext, resolve_context,
    ARRAY, RUN, RUN_ARRAY, EVENT_ARRAY, STRING,
)
from labquake_explorer.ui.actions import Action, actions_for, register_action
# Importing the views package registers every view's context-menu action.
from labquake_explorer.ui.views import (
    SimplePlotView, PointsSelectorView, IndexPickerView, SlopeAnalyzerView, RunSignalsView,
)


class LabquakeExplorer:
    def __init__(self, root: tk.Tk):
        self.config = LabquakeExplorerConfig()
        self.root = root
        self.root.title(f"{self.config.WINDOW_TITLE} {__version__}")
        
        self.data_manager = DataManager()
        self.child_windows: List[tk.Toplevel] = []
        self.data_tree: Optional[ttk.Treeview] = None
        self.active_context_menu: Optional[tk.Menu] = None
        self.current_file_path: Optional[Path] = None
        
        self.setup_window()
        self.create_widgets()
        self.setup_bindings()

    def setup_window(self) -> None:
        screen_height = self.root.winfo_screenheight()
        window_height = screen_height - (self.config.WINDOW_GAP * 3)

        self.set_window_icon(self.root)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.geometry(
            f"{self.config.WINDOW_WIDTH}x{window_height}+"
            f"{self.config.WINDOW_GAP}+{self.config.WINDOW_GAP}"
        )
        self.root.lift()
        self.root.focus_force()

    def create_widgets(self) -> None:
        self.create_buttons()
        self.init_data_tree()

    def create_buttons(self) -> None:
        buttons = [
            ("Load", self.load_file, "normal", 0),
            ("Refresh", self.refresh_tree, "normal", 1),
            ("Save As", self.save_file, "disabled", 3)
        ]
        
        for text, command, state, col in buttons:
            btn = tk.Button(self.root, text=text, command=command, state=state)
            btn.grid(row=1, column=col, padx=2, pady=2, sticky="w" if col < 2 else "e")
            if text == "Save As":
                self.save_button = btn

    def init_data_tree(self) -> None:
        if self.data_tree:
            self.data_tree.destroy()
            
        self.data_tree = ttk.Treeview(self.root)
        self.data_tree.grid(row=0, column=0, columnspan=4, padx=2, pady=2, sticky="nsew")
        header_text = self.current_file_path.name if self.current_file_path else "[Data File]"
        self.data_tree.heading("#0", text=header_text, anchor="w")
        
        self.data_tree.bind("<Double-1>", self.on_double_click)
        self.data_tree.bind("<Button-1>", self.on_left_click)
        self.data_tree.bind("<Button-2>", self.on_right_click)
        self.data_tree.bind("<Button-3>", self.on_right_click)

    def load_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=self.config.FILE_TYPES
        )
        if not file_path:
            return

        try:
            path = Path(file_path)
            self.data_manager.load_file(path)
            self.current_file_path = path
            self.save_button.configure(state="normal")
            self.refresh_tree()
            print(f"File loaded: {file_path}")

            # Expand the runs node
            for item in self.data_tree.get_children(""):
                if self.data_tree.item(item)["text"].startswith("runs"):
                    self.data_tree.item(item, open=True)
                    break
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_file(self) -> None:
        current = self.current_file_path
        initial_file = None
        if current:
            keep = current.suffix.lower() in (".h5", ".hdf5")
            initial_file = current.name if keep else current.with_suffix(self.config.SAVE_SUFFIX).name

        file_path = filedialog.asksaveasfilename(
                title="Save data file",
                initialfile=initial_file,
                initialdir=str(current.parent) if current else None,
                defaultextension=self.config.SAVE_SUFFIX,
                filetypes=self.config.FILE_TYPES,
            )
        if not file_path:
            return

        try:
            self.data_manager.save_file(Path(file_path))
            print(f"File saved: {file_path}")
            self.current_file_path = Path(file_path)
            self.refresh_tree()
            messagebox.showinfo("Success", "File saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def refresh_tree(self) -> None:
        if not self.data_manager.data:
            return
            
        selected_item = self.data_tree.selection()[0] if self.data_tree.selection() else None
        self.init_data_tree()
        self.build_tree(self.data_manager.data, "")
        
        if selected_item:
            try:
                self.data_tree.focus(selected_item)
                self.data_tree.selection_set(selected_item)
                self.data_tree.see(selected_item)
            except:
                pass

    def build_tree(self, data: Dict[str, Any], parent_iid: str) -> None:
        """Recursively build tree view from data"""
        if isinstance(data, dict):
            for key, value in data.items():
                label = self.format_tree_label(key, value)
                iid = self.data_tree.insert(parent_iid, "end", text=label)
                if isinstance(value, (dict, list)):
                    self.build_tree(value, iid)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                try:
                    label = f"[{i}]: {value['name']}"
                except:
                    label = self.format_tree_label(f"[{i}]", value)
                iid = self.data_tree.insert(parent_iid, "end", text=label)
                if isinstance(value, (dict, list)):
                    self.build_tree(value, iid)

    def format_tree_label(self, key: str, value: Any) -> str:
        """Format label for tree view items based on data type and content.

        Args:
            key: The key or index name
            value: The value to format

        Returns:
            Formatted string label
        """
        if isinstance(value, str):
            return f"{key}: {value}"
        elif isinstance(value, (int, float, np.floating, np.integer)):
            return f"{key}: {value}"
        elif is_channel_array(value):
            n_ch, n = value["data"].shape
            unit = f" {value['unit']}" if value.get("unit") else ""
            return f"{key}: {n_ch} channels x {n}{unit}"
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return f"{key}: {value.flatten()[0]}"
            else:
                shape_str = str(list(value.shape)).replace(" ", "")  # Remove spaces
                return f"{key}: array{shape_str}"
        elif isinstance(value, list):
            if len(value) == 1 and not key == 'events':
                return f"{key}: {value[0]}"
            else:
                return f"{key}: array[{len(value)}]"
        return f"{key}: {type(value).__name__}"
    
    def get_full_path(self, item=None):
        """``(path, key)`` of a tree item.  Tree paths are data paths, not file
        paths: they are always ``/``-joined (``os.path.join`` would give
        backslashes on Windows, which the data layer and the context resolver
        do not use)."""
        def clean_up_text(s):
            return s.split(':')[0].strip()
        if item is None:
            item = self.data_tree.selection()[0]
        parent_iid = self.data_tree.parent(item)
        node = []
        # go backward until reaching root
        while parent_iid != '':
            node.insert(0, clean_up_text(self.data_tree.item(parent_iid)['text']))
            parent_iid = self.data_tree.parent(parent_iid)
        i = clean_up_text(self.data_tree.item(item, "text"))
        return "/".join([*node, i]), i

    def find_item(self, target_path: str, item: str = "") -> Optional[str]:
        """Find the tree item id whose full path equals ``target_path``."""
        for child in self.data_tree.get_children(item):
            if self.get_full_path(child)[0] == target_path:
                return child
            found = self.find_item(target_path, child)
            if found:
                return found
        return None

    def context_at(self, item=None) -> TreeContext:
        """Resolve the selected (or given) tree item into a TreeContext."""
        path, _ = self.get_full_path(item)
        try:
            value = self.data_manager.get_data(path)
        except Exception:
            value = None
        return resolve_context(path, value)

    # ------------------------------------------------------------------
    # child window bookkeeping
    # ------------------------------------------------------------------
    def register_child(self, view: tk.Toplevel) -> None:
        """Track a child window: set its icon and keep it for cleanup."""
        self.set_window_icon(view)
        if view not in self.child_windows:
            self.child_windows.append(view)

    def unregister_child(self, view: tk.Toplevel) -> None:
        while view in self.child_windows:
            self.child_windows.remove(view)

    def run_action(self, action: Action, ctx: TreeContext) -> None:
        try:
            action.run(self, ctx)
        except Exception as e:
            messagebox.showerror("Error", f"{action.label} failed: {e}")
            raise

    # ------------------------------------------------------------------
    # context-menu actions that are commands rather than views
    # ------------------------------------------------------------------
    @register_action("Min/Max", kinds=[EVENT_ARRAY], order=30)
    def min_max(self, ctx: TreeContext) -> None:
        path = ctx.path
        y = self.data_manager.get_data(path)
        x = np.arange(len(y))
        values = np.asarray(y, dtype=float)
        if not np.isfinite(values).any():
            messagebox.showwarning("Min/Max", f"{path} has no finite samples")
            return
        # NaN samples are never the extreme (np.argmin/argmax would return the first NaN)
        idx_min = int(np.nanargmin(values))
        idx_max = int(np.nanargmax(values))
        picked_idx = [idx_max, idx_min]
        view = PointsSelectorView(self, x, y, picked_idx, add_remove_enabled=False,
                                 xlabel='index', ylabel=ctx.key, title=path)
        self.register_child(view)

    @register_action("Pick Indices", kinds=[ARRAY, RUN_ARRAY, EVENT_ARRAY], order=20)
    def pick_indices(self, ctx: TreeContext) -> None:
        view = IndexPickerView(self, item_y=ctx.path)
        self.register_child(view)

    @register_action("Extract Slopes", kinds=[ARRAY, RUN_ARRAY, EVENT_ARRAY], order=21)
    def extract_slope(self, ctx: TreeContext) -> None:
        view = SlopeAnalyzerView(self, item_y=ctx.path)
        self.register_child(view)

    @register_action("Edit String", kinds=[STRING], order=10)
    def edit_string(self, ctx: TreeContext) -> None:
        """Edit a string value in the data structure"""
        path = ctx.path
        data = self.data_manager.get_data(path)

        new_string = simpledialog.askstring('Edit String', f'{path}', initialvalue=data)
        if new_string is None:
            print('Edit String aborted.')
            return
        
        self.data_manager.set_data(path, new_string)
        self.refresh_tree()

    # ------------------------------------------------------------------
    # tree interaction
    # ------------------------------------------------------------------
    def on_double_click(self, event):
        path, item = self.get_full_path()
        print(f"Double-clicked on item: {path}")
        data = self.data_manager.get_data(path)
        ctx = resolve_context(path, data)
        if ctx.kind == RUN:
            RunSignalsView(self, ctx.run_idx)
        elif type(data) is np.ndarray:
            print(f"plotting {item}")
            view = SimplePlotView(self)
            view.ax.plot(data)
            view.ax.set_xlabel('index')
            view.ax.set_ylabel(item)
            view.ax.set_title(path.replace('/[', '['))
            view.canvas.draw_idle()
        elif type(data) is dict:
            print('dict')
        elif type(data) is list:
            print('list')
        else:
            print(data)
        
    def on_left_click(self, event):
        if self.active_context_menu:
            self.active_context_menu.unpost()

    def build_context_menu(self, ctx: TreeContext) -> Optional[tk.Menu]:
        """Build a context menu from the registered actions for ``ctx.kind``."""
        actions = actions_for(ctx.kind)
        if not actions:
            return None
        menu = tk.Menu(self.root, tearoff=0)
        for action in actions:
            menu.add_command(label=action.label,
                             command=lambda a=action: self.run_action(a, ctx))
        return menu

    def on_right_click(self, event):
        # Clear previous menu (unpost AND destroy: a fresh Menu is built per
        # right-click, and orphaned menus would otherwise accumulate under root)
        if self.active_context_menu:
            self.active_context_menu.unpost()
            try:
                self.active_context_menu.destroy()
            except tk.TclError:
                pass
        self.active_context_menu = None

        try:
            item = self.data_tree.selection()[0]
        except:
            item = None
        if not item:
            return

        ctx = self.context_at(item)
        self.active_context_menu = self.build_context_menu(ctx)
        if self.active_context_menu:
            self.active_context_menu.post(event.x_root, event.y_root)
    
    def has_child_name_contains(self, item_id, keyword):
        if not item_id:
            return False
        children = self.data_tree.get_children(item_id)
        return any(keyword in self.data_tree.item(child_id, "text") for child_id in children)
    
    def has_child_named(self, item_id, name):
        if not item_id:
            return False
        children = self.data_tree.get_children(item_id)
        return any(name == self.data_tree.item(child_id, "text").split(":")[0] for child_id in children)

    def on_delete(self, event) -> None:
        """Handle deletion of items from the tree view"""
        try:
            item_id = self.data_tree.selection()[0]

            # Get the item above the selected item
            prev_item = self.data_tree.prev(item_id)
            if not prev_item:
                # If no previous item, get the parent
                prev_item = self.data_tree.parent(item_id)

            # Store the path of the item to focus
            focus_path = self.get_full_path(prev_item)[0] if prev_item else ""

            item_path = self.get_full_path(item_id)[0]

            # Confirm deletion with user
            ans = messagebox.askokcancel(
                title="Confirmation", 
                message=f'This procedure will delete "{item_path}".', 
                icon=messagebox.WARNING
            )
            if not ans:
                return

            # Attempt deletion
            try:
                self.data_manager.delete_data(item_path)
                self.refresh_tree()

                # After refresh, find and focus the previous item
                if focus_path:
                    for item in self.data_tree.get_children(""):
                        if self._find_and_focus_item(item, focus_path):
                            break

            except (ValueError, KeyError, IndexError) as e:
                messagebox.showerror("Error", f"Failed to delete item: {str(e)}")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def _find_and_focus_item(self, item: str, target_path: str) -> bool:
        """Helper method to find and focus an item by its path

        Args:
            item: The current tree item ID to check
            target_path: The path to find

        Returns:
            bool: True if item was found and focused
        """
        current_path = self.get_full_path(item)[0]
        if current_path == target_path:
            self.data_tree.focus(item)
            self.data_tree.selection_set(item)
            self.data_tree.see(item)
            return True

        # Recursively check children
        for child in self.data_tree.get_children(item):
            if self._find_and_focus_item(child, target_path):
                return True

        return False

    def on_closing(self) -> None:
        try:
            # First withdraw (hide) all windows
            for window in self.child_windows[:]:
                if window.winfo_exists():
                    window.withdraw()
            self.root.withdraw()
            
            # Then destroy them
            for window in self.child_windows[:]:
                if window.winfo_exists():
                    window.destroy()
                self.child_windows.remove(window)
                
            self.root.destroy()
            sys.exit(0)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            sys.exit(1)

    def setup_bindings(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Delete>", self.on_delete)

    def set_window_icon(self, window: tk.Tk | tk.Toplevel) -> None:
        """Set the application icon for any window.

        Args:
            window: The window (root or Toplevel) to set the icon for
        """
        # Go up three levels from the current file to reach project root
        project_root = Path(__file__).parent.parent.parent
        icon_path = project_root / "assets" / "icons" / "labquake_explorer"

        try:
            if sys.platform == "darwin":  # macOS
                img = tk.Image("photo", file=str(icon_path.with_suffix(".png")))
                window.tk.call('wm', 'iconphoto', window._w, img)
            elif sys.platform == "win32":  # Windows
                window.iconbitmap(str(icon_path.with_suffix(".ico")))
            elif sys.platform.startswith("linux"):  # Linux
                img = tk.PhotoImage(file=str(icon_path.with_suffix(".png")))
                window.tk.call('wm', 'iconphoto', window._w, img)
        except Exception as e:
            print(f"Error setting icon for window: {e}")
