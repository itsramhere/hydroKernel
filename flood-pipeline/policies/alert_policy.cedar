// Statutory Criteria for Automated Disaster Broadcast
permit (
    principal == Role::"MLHydrologyAgent",
    action == Action::"BroadcastAlert",
    resource
)
when {
    context.flood_depth_cm >= 15 &&
    context.time_to_peak_hours <= 6 &&
    context.affected_parcels_count > 0
};

// Safeguard: Explicitly forbid broadcasts in DRY_RUN mode
forbid (
    principal,
    action,
    resource
)
when {
    context.system_mode == "DRY_RUN"
};
