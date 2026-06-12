"""Real Firefox probe names from gecko-dev master, audited 2026-04-29.
Sources: browser/components/metrics.yaml, toolkit/components/telemetry/Histograms.json.
Per-flavor fill probabilities are synthetic (Mozilla doesn't publish them)."""

# 64 real Firefox probe names (Glean + Histograms), grouped by category.
PROBE_NAMES = [
    # ----- browser.startup / session ---------------------------------
    "browser.launched_to_handle.system_notification",
    "browser.startup.abouthome_cache_result",
    "browser.startup.abouthome_cache_shutdownwrite",
    "browser.startup.kiosk_mode",
    "startup.is_cold",
    "startup.seconds_since_last_os_restart",
    "os.environment.launch_method",
    "os.environment.launched_to_handle",
    "os.environment.invoked_to_handle",
    "os.environment.is_default_handler",
    "os.environment.is_kept_in_dock",
    "os.environment.is_taskbar_pinned",
    "os.environment.is_taskbar_pinned_private",
    # ----- security / privacy controls -------------------------------
    "security.https_only_mode_enabled",
    "security.https_only_mode_enabled_pbm",
    "security.global_privacy_control_enabled",
    "primary.password.enabled",
    "sslkeylogging.enabled",
    # ----- data sanitisation -----------------------------------------
    "datasanitization.privacy_sanitize_sanitize_on_shutdown",
    "datasanitization.privacy_clear_on_shutdown_cookies",
    "datasanitization.privacy_clear_on_shutdown_history",
    "datasanitization.privacy_clear_on_shutdown_formdata",
    "datasanitization.privacy_clear_on_shutdown_downloads",
    "datasanitization.privacy_clear_on_shutdown_cache",
    "datasanitization.privacy_clear_on_shutdown_sessions",
    "datasanitization.privacy_clear_on_shutdown_offline_apps",
    "datasanitization.privacy_clear_on_shutdown_site_settings",
    "datasanitization.privacy_clear_on_shutdown_open_windows",
    "datasanitization.session_permission_exceptions",
    # ----- enterprise / launch / default-browser ---------------------
    "background_update.reasons_to_not_update",
    "background_update.time_last_update_scheduled",
    "start_menu.manually_unpinned_since_last_launch",
    "launch_on_login.last_profile_disable_startup",
    "upgrade_dialog.trigger_reason",
    "browser.is_user_default",
    "browser.is_user_default_error",
    "browser.set_default_dialog_prompt_rawcount",
    "browser.set_default_always_check",
    "browser.set_default_result",
    "browser.set_default_error",
    "browser.default_at_launch",
    "browser.attribution_errors",
    # ----- legacy histograms (developer / memory) --------------------
    "MEMORY_RESIDENT_FAST",
    "MEMORY_TOTAL",
    "MEMORY_UNIQUE",
    "MEMORY_VSIZE",
    "MEMORY_HEAP_ALLOCATED",
    "MEMORY_FREE_PURGED_PAGES_MS",
    "GHOST_WINDOWS",
    "CYCLE_COLLECTOR",
    "CYCLE_COLLECTOR_FULL",
    "CYCLE_COLLECTOR_MAX_PAUSE",
    "CYCLE_COLLECTOR_TIME_BETWEEN",
    "GC_IN_PROGRESS_MS",
    "GC_SLICE_DURING_IDLE",
    "GPU_PROCESS_LAUNCH_TIME_MS_2",
    "GPU_PROCESS_INITIALIZATION_TIME_MS",
    "CHILD_PROCESS_LAUNCH_MS",
    "CONTENT_PROCESS_LAUNCH_TOTAL_MS",
    "CHECKERBOARD_DURATION",
    "DEVICE_RESET_REASON",
    "FORCED_DEVICE_RESET_REASON",
    "FULLSCREEN_TRANSITION_BLACK_MS",
    "FIREFOX_VIEW_CUMULATIVE_SEARCHES",
    "PAGE_FAULTS_HARD",
    "LOW_MEMORY_EVENTS_PHYSICAL",
]

assert len(PROBE_NAMES) == 66, len(PROBE_NAMES)


# Per-flavor probe-fill probabilities. Each flavor is a plausible
# Firefox-Desktop configuration; numbers reflect how likely a probe in
# that category is to be filled in given the configuration's defaults
# and feature flags.
#
# Flavors:
#   F0  Default release          (50%)
#   F1  Privacy-focused           (25%) -- HTTPS-only, sanitize-on-shutdown,
#                                          GPC, no enterprise probes
#   F2  Enterprise/managed        (15%) -- kiosk, launch-on-login,
#                                          set-default flow active
#   F3  Developer/Beta            (8%)  -- memory and CC histograms
#                                          present, devtools enabled
#   F4  Nightly + research arm    (2%)  -- experimental probes ON,
#                                          additional debug
FLAVORS = [
    "F0 default-release",
    "F1 privacy-focused",
    "F2 enterprise/managed",
    "F3 developer/beta",
    "F4 nightly + research",
]
PREVALENCE = [0.50, 0.25, 0.15, 0.08, 0.02]


# Per-flavor fill rate for each category prefix. Order: F0..F4.
CATEGORY_RATES = {
    # Glean lower-case categories.
    "browser":              [0.92, 0.88, 0.95, 0.85, 0.90],
    "startup":              [0.92, 0.85, 0.90, 0.88, 0.92],
    "os":                   [0.90, 0.88, 0.95, 0.85, 0.92],
    "security":             [0.30, 0.95, 0.20, 0.40, 0.95],
    "primary":              [0.05, 0.45, 0.20, 0.10, 0.10],
    "sslkeylogging":        [0.001, 0.001, 0.001, 0.30, 0.40],
    "datasanitization":     [0.10, 0.85, 0.15, 0.20, 0.30],
    "background_update":    [0.40, 0.30, 0.95, 0.50, 0.50],
    "start_menu":           [0.50, 0.30, 0.95, 0.40, 0.50],
    "launch_on_login":      [0.10, 0.05, 0.95, 0.20, 0.20],
    "upgrade_dialog":       [0.50, 0.40, 0.20, 0.60, 0.70],
    # Legacy histograms, upper-case prefixes.
    "MEMORY":               [0.50, 0.40, 0.55, 0.95, 0.97],
    "CYCLE":                [0.50, 0.40, 0.40, 0.95, 0.97],
    "GC":                   [0.50, 0.40, 0.40, 0.95, 0.97],
    "GPU":                  [0.55, 0.40, 0.45, 0.95, 0.97],
    "CHILD":                [0.50, 0.40, 0.45, 0.95, 0.95],
    "CONTENT":              [0.55, 0.45, 0.55, 0.92, 0.95],
    "CHECKERBOARD":         [0.30, 0.20, 0.30, 0.85, 0.92],
    "DEVICE":               [0.10, 0.05, 0.05, 0.70, 0.80],
    "FORCED":               [0.05, 0.02, 0.02, 0.50, 0.65],
    "FULLSCREEN":           [0.40, 0.20, 0.30, 0.85, 0.90],
    "FIREFOX":              [0.65, 0.55, 0.55, 0.85, 0.90],
    "PAGE":                 [0.40, 0.30, 0.40, 0.85, 0.92],
    "LOW":                  [0.10, 0.05, 0.10, 0.70, 0.85],
    "GHOST":                [0.10, 0.05, 0.10, 0.85, 0.92],
}


def _category_of(name: str) -> str:
    if "." in name:
        return name.split(".")[0]
    return name.split("_")[0]


def build_profile_matrix():
    import numpy as np
    rows = []
    for f_idx in range(len(FLAVORS)):
        row = []
        for name in PROBE_NAMES:
            cat = _category_of(name)
            rates = CATEGORY_RATES.get(cat)
            if rates is None:
                raise KeyError(f"No CATEGORY_RATES for {cat!r} (probe {name!r})")
            row.append(rates[f_idx])
        rows.append(row)
    return np.asarray(rows)


if __name__ == "__main__":
    print(f"Loaded {len(PROBE_NAMES)} real Firefox probe names "
          f"across {len(FLAVORS)} flavors.")
    M = build_profile_matrix()
    print("Profile matrix shape:", M.shape)
    print("Per-flavor mean fill rate:", [f"{p:.2f}" for p in M.mean(axis=1)])
