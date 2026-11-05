# Stop every process in the Tier-1 GPU queue. Called by scripts/stop_all.sh.
#
# WHY THIS IS NOT JUST "kill the lock holders" (bug found live 2026-08-13):
# the queue is THREE levels deep —
#
#   tier1_chain.sh  ->  <sub>_batch.sh  ->  python run_intervention.py
#
# and only the TOP level writes a lock file. Killing the lock holder with -9
# leaves the sub-batch shell running as an orphan, and that shell is the thing
# that launches trainers — so the GPU stayed busy after a "successful" stop,
# with new arms starting minutes later. Kill the whole tree, by pattern.
#
# ORDER: shells before trainers, always. Killing a trainer first just makes its
# parent shell launch the next one. Killing the shells first removes the spawner,
# and only then is killing the trainer final.
#
# THE RACE: a shell can spawn a trainer microseconds before it dies, leaving an
# orphaned trainer with a dead parent (observed: pid 34936, parent already gone).
# One pass cannot close this, so we sweep until two consecutive passes find
# nothing, up to $maxPasses.

$repo = Split-Path -Parent $PSScriptRoot
$shellPat = 'tier1_chain\.sh|rate_collapse_60M_batch\.sh|intervention_arms_seeds23_batch\.sh|wsd_schedule_batch\.sh|lr_decomposition_batch\.sh'

function Get-QueueShells {
    Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
        Where-Object { $_.CommandLine -match $shellPat -and $_.CommandLine -notmatch 'CimInstance|stop_all' }
}
function Get-QueueTrainers {
    # Matched on the REPO'S OWN venv interpreter, so a system or store python
    # (and any unrelated venv) is never touched.
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($repo, [StringComparison]::OrdinalIgnoreCase) }
}

$maxPasses = 6
$clean = 0
for ($i = 1; $i -le $maxPasses; $i++) {
    $shells   = @(Get-QueueShells)
    $trainers = @(Get-QueueTrainers)

    if ($shells.Count -eq 0 -and $trainers.Count -eq 0) {
        $clean++
        # Two consecutive empty passes: nothing spawned in the gap, so the
        # spawner really is gone rather than merely between launches.
        if ($clean -ge 2) { Write-Host "   pass ${i}: clear (confirmed twice)"; break }
        Start-Sleep -Milliseconds 700
        continue
    }
    $clean = 0

    foreach ($s in $shells) {
        Write-Host ("   pass ${i}: killing shell   " + $s.ProcessId + "  " + ($s.CommandLine -replace '.*bash\.exe"?\s*',''))
        Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue
    }
    foreach ($t in $trainers) {
        $c = $t.CommandLine -replace '\s+',' '
        Write-Host ("   pass ${i}: killing trainer " + $t.ProcessId + "  " + $c.Substring(0, [Math]::Min(110, $c.Length)))
        Stop-Process -Id $t.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 700
}

$sLeft = @(Get-QueueShells).Count
$tLeft = @(Get-QueueTrainers).Count
if ($sLeft -eq 0 -and $tLeft -eq 0) {
    Write-Host "   RESULT: no queue shells, no trainers running"
    exit 0
} else {
    Write-Host "   RESULT: STILL ALIVE after $maxPasses passes - shells=$sLeft trainers=$tLeft"
    Write-Host "   Something is respawning faster than the sweep; investigate before relaunching."
    exit 1
}
