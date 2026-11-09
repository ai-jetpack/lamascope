#pragma once

bool IsProcessElevated();
// Relaunches this exe with the "runas" verb. Returns true if the new process started.
bool RestartAsAdmin();
