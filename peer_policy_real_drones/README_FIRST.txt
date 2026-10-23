Peer 4-drone policy (AS-Tello-Room-Peer-v0, peer_qualified_84M) + the real-drone runner.
Code is from commit 42a46db (the runner was removed from main in a3527a1).

Policy:  simulation_infra/policies/peer_qualified_84M/pt/policy.pt  (+ model_vecnormalize.pkl)
Runner:  drone_deployment_infra/swarm_ops/swarm_policy.py

  cd peer_policy_real_drones/drone_deployment_infra/swarm_ops   (from the repo root)
  pip install -r requirements.txt torch
  python swarm_policy.py --checkpoint ../../simulation_infra/policies/peer_qualified_84M/pt/policy.pt \
      --drones TELLO-3,TELLO-4,TELLO-5,TELLO-1          # dry-run, nothing flies
  ... add --fly to really fly.

WARNING: the policy was trained on the OLD room map (tags 5-10). The room was re-mapped to
tags 8,9,10 on 2026-10-17, so check with the colleague before flying. Sim-qualified only.

Not in git: config/dongles.json (dongle <-> drone mapping, machine-specific). Copy yours into
drone_deployment_infra/config/ before running.
