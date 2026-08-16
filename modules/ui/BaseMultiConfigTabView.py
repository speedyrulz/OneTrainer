from abc import ABC, abstractmethod

from modules.util.enum.MultiConfigMode import MultiConfigMode
from modules.util.enum.MultiConfigSelection import MultiConfigSelection
from modules.util.enum.MultiConfigStateStorage import MultiConfigStateStorage


class BaseMultiConfigTabView(ABC):
    """Layout for the Multi Config tab, shared by both UI frameworks.

    The tab has three parts. The top is fixed. The middle depends on the mode and is rebuilt when it
    changes. The value inputs inside the middle depend on the swept setting and how many values were
    asked for, and are rebuilt again when either changes - which is why they live in their own frame
    rather than beside the dropdowns that trigger the rebuild.
    """

    def __init__(self, components):
        self.components = components
        # options_kv fires its command once while it is being created, to push the starting value out.
        # Acting on that would tear down the frame currently being built, so refreshes are ignored
        # until the build finishes.
        self._building = False

    @abstractmethod
    def refresh_mode_frame(self):
        """Discard and rebuild the mode-dependent middle section."""

    @abstractmethod
    def refresh_values_frame(self):
        """Discard and rebuild the value inputs."""

    def request_mode_refresh(self):
        if not self._building:
            self.refresh_mode_frame()

    def request_values_refresh(self):
        if not self._building:
            self.refresh_values_frame()

    # --- fixed section --------------------------------------------------------------------------

    def build_header(self, master, controller, ui_state):
        self.components.label(
            master, 0, 0, "Multi-Config Training",
            tooltip="Train several settings against each other. Each one trains for one validation "
                    "interval from the same starting point, and the one with the best validation loss "
                    "continues into the next interval. Requires validation to be enabled on the "
                    "general tab, and at least one concept marked as a validation concept.",
            wide_tooltip=True,
        )
        self.components.switch(master, 0, 1, ui_state, "multi_config")

        self.components.label(
            master, 1, 0, "Mode",
            tooltip="Full configs races up to 10 saved config files, which may differ in many "
                    "settings at once. Single setting races up to 10 values of one setting, taking "
                    "everything else from the training page so the comparison isolates one variable. "
                    "Adaptive learning rate races a window of learning rates every round and follows "
                    "the winner, so the learning rate moves during the run.",
            wide_tooltip=True,
        )
        self.components.options_kv(
            master, 1, 1, controller.get_modes(), ui_state, "multi_config_mode",
            command=lambda _value: self.request_mode_refresh(),
        )

    def build_footer(self, master, controller, ui_state):
        row = 0

        self.components.label(
            master, row, 0, "End Early",
            tooltip="Stop training once the number of rounds set beside this fail in a row to reach a "
                    "lower validation loss than the best one seen so far. The model saved at the end "
                    "is the best one either way, so ending early only saves the time that would have "
                    "been spent getting worse.",
            wide_tooltip=True,
        )
        self.components.switch(master, row, 1, ui_state, "multi_config_end_early")

        self.components.label(
            master, row, 2, "Rounds Without Improvement",
            tooltip="How many rounds in a row must fail to beat the best validation loss before "
                    "training stops.",
            wide_tooltip=True,
        )
        self.components.entry(master, row, 3, ui_state, "multi_config_end_early_rounds")
        row += 1

        self.components.label(
            master, row, 0, "Drop Losing Configs",
            tooltip="Every so many rounds, drop the configs that have not won any of them. A config "
                    "that keeps losing costs a full segment of training per round for a result the "
                    "run will not use. The last config standing is never dropped, and training then "
                    "continues normally. Does not apply to the adaptive learning rate mode, where "
                    "every round compares a fresh set of rates.",
            wide_tooltip=True,
        )
        self.components.switch(master, row, 1, ui_state, "multi_config_prune")

        self.components.label(
            master, row, 2, "Check Every",
            tooltip="How many validation rounds to watch before dropping the configs that won none "
                    "of them.",
            wide_tooltip=True,
        )
        self.components.entry(master, row, 3, ui_state, "multi_config_prune_after")
        row += 1

        self.components.label(
            master, row, 0, "Drop Rule",
            tooltip="Never won: drop every config that won none of the rounds in the window. Clears "
                    "several at once, but drops a config that is consistently second by a hair just "
                    "as readily as one that is nowhere near. Worst average loss: drop the single "
                    "config with the highest average validation loss over the window. Slower to "
                    "narrow the field, but it ranks on how far behind a config actually is.",
            wide_tooltip=True,
        )
        self.components.options_kv(
            master, row, 1, controller.get_prune_strategies(), ui_state, "multi_config_prune_strategy",
        )
        row += 1

        self.components.label(
            master, row, 0, "Selection Metric",
            tooltip="How the validation losses are reduced to the single number the winner is picked "
                    "on. TOTAL_AVERAGE averages over every validation sample. MEAN_PER_CONCEPT "
                    "averages the per-concept losses so a small concept counts as much as a large "
                    "one. WORST_CONCEPT picks the settings that leave no concept behind.",
            wide_tooltip=True,
        )
        self.components.options(
            master, row, 1, [str(x) for x in list(MultiConfigSelection)], ui_state, "multi_config_selection"
        )
        row += 1

        self.components.label(
            master, row, 0, "State Storage",
            tooltip="Where the per-config training states are kept between rounds. DISK writes them "
                    "to <workspace>/multi_config/state. RAM is faster but needs enough free system "
                    "memory to hold one copy of the trainable weights and optimizer state per config.",
            wide_tooltip=True,
        )
        self.components.options(
            master, row, 1, [str(x) for x in list(MultiConfigStateStorage)], ui_state,
            "multi_config_state_storage"
        )
        row += 1

        self.components.label(
            master, row, 0, "Save Each Round",
            tooltip="Save the winning model after every round, in addition to the final save",
        )
        self.components.switch(master, row, 1, ui_state, "multi_config_save_each_round")
        row += 1

        self.components.label(
            master, row, 0, "Deterministic Data",
            tooltip="Rebuild the data pipeline for each config so every config in a round trains on "
                    "exactly the same batches and validates on exactly the same samples. Turning "
                    "this off skips the rebuild on epoch boundaries, which is faster but means the "
                    "configs are compared on different shuffles of the data.",
            wide_tooltip=True,
        )
        self.components.switch(master, row, 1, ui_state, "multi_config_deterministic_data")

    # --- mode-dependent section -----------------------------------------------------------------

    def build_mode_content(self, master, controller, ui_state) -> int:
        """Fill the middle section. Returns the next free row, where the values frame goes."""
        mode = controller.config.multi_config_mode
        if mode == MultiConfigMode.ADAPTIVE_LEARNING_RATE:
            return self.__build_adaptive_header(master, controller, ui_state)
        if mode == MultiConfigMode.SINGLE_SETTING:
            return self.__build_sweep_header(master, controller, ui_state)
        return self.__build_config_slots(master, controller, ui_state)

    def __build_adaptive_header(self, master, controller, ui_state) -> int:
        self.components.label(
            master, 1, 0, "Number of Learning Rates",
            tooltip="How many rates each round compares. The window moves further when the winner is "
                    "further from its middle, so a wider window finds a distant learning rate in "
                    "fewer rounds but costs a full segment of training per extra rate.",
            wide_tooltip=True,
        )
        self.components.options(
            master, 1, 1, controller.get_adaptive_counts(), ui_state, "multi_config_adaptive_count",
            command=lambda _value: self.request_values_refresh(),
        )

        self.components.label(
            master, 0, 0, "Starting Learning Rate",
            tooltip="Where the ladder starts. It sits in the middle of the first round's window. "
                    "Rates move the leading digit and roll over at each decade, so 0.0001 steps down "
                    "to 0.00009 and 0.0009 steps up to 0.001. A value that is not a single leading "
                    "digit is snapped to the nearest rung.",
            wide_tooltip=True,
        )
        self.components.entry(
            master, 0, 1, ui_state, "multi_config_adaptive_lr", required=True,
            command=self.request_values_refresh,
        )
        return 2

    def __build_config_slots(self, master, controller, ui_state) -> int:
        self.components.label(
            master, 0, 0, "Number of Configs",
            tooltip="How many config files to race. Two or more are needed for the tournament to "
                    "have anything to compare. Slots past this number are ignored.",
            wide_tooltip=True,
        )
        self.components.options(
            master, 0, 1, controller.get_path_counts(), ui_state, "multi_config_path_count",
            command=lambda _value: self.request_values_refresh(),
        )
        return 1

    def __build_sweep_header(self, master, controller, ui_state) -> int:
        self.components.label(
            master, 0, 0, "Setting",
            tooltip="Which training setting to vary. Everything else comes from the training page.",
        )
        self.components.options_kv(
            master, 0, 1, controller.get_sweep_settings(), ui_state, "multi_config_sweep_setting",
            command=lambda _value: self.request_values_refresh(),
        )

        self.components.label(
            master, 1, 0, "Number of Values",
            tooltip="How many values of that setting to compare. Two or more are needed for the "
                    "tournament to have anything to compare.",
        )
        self.components.options(
            master, 1, 1, controller.get_sweep_counts(), ui_state, "multi_config_sweep_count",
            command=lambda _value: self.request_values_refresh(),
        )

        notice = controller.sweep_notice()
        if notice:
            self.components.label(master, 2, 0, "Note")
            self.components.label(master, 2, 1, notice, wraplength=420, tooltip=notice,
                                  wide_tooltip=True)
            return 3

        return 2

    # --- value inputs ---------------------------------------------------------------------------

    def build_value_inputs(self, master, controller, ui_state):
        mode = controller.config.multi_config_mode

        if mode == MultiConfigMode.ADAPTIVE_LEARNING_RATE:
            self.components.label(master, 0, 0, "First Round")
            self.components.label(
                master, 0, 1, controller.ladder_preview(),
                tooltip="The learning rates the first round will compare. Later rounds follow "
                        "whichever one wins.",
                wide_tooltip=True,
            )
            return

        if mode == MultiConfigMode.FULL_CONFIGS:
            for slot in range(1, controller.path_count + 1):
                self.components.label(
                    master, slot - 1, 0, f"Config Set {slot}",
                    tooltip="A config file saved with 'Save config'. Only the settings a config is "
                            "allowed to vary are taken from it (learning rate and schedule, "
                            "optimizer, loss weights, noise and timestep settings); the model, "
                            "dataset and every other structural setting comes from the config open "
                            "right now. Leave empty to skip this slot. At least 2 must be filled.",
                    wide_tooltip=True,
                )
                self.components.path_entry(
                    master, slot - 1, 1, ui_state, f"multi_config_path_{slot}",
                    mode="file", allow_model_files=False,
                )
            return

        if mode != MultiConfigMode.SINGLE_SETTING:
            return

        label = controller.sweep_label()
        tooltip = controller.sweep_tooltip()
        is_choice = controller.sweep_is_choice()
        choices = controller.sweep_choices() if is_choice else []

        for slot in range(1, controller.sweep_count + 1):
            var_name = f"multi_config_sweep_value_{slot}"
            self.components.label(master, slot - 1, 0, f"{label} {slot}", tooltip=tooltip,
                                  wide_tooltip=True)
            if is_choice:
                self.components.options(master, slot - 1, 1, choices, ui_state, var_name)
            else:
                self.components.entry(master, slot - 1, 1, ui_state, var_name, tooltip=tooltip,
                                      required=True)

    def apply_repaired_sweep_values(self, controller, ui_state):
        """Replace slot contents that do not fit the selected setting before the inputs are built."""
        if controller.config.multi_config_mode != MultiConfigMode.SINGLE_SETTING:
            return
        for var_name, value in controller.repair_sweep_values().items():
            ui_state.get_var(var_name).set(value)
