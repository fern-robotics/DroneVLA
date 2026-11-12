from integrations.openvla.prompt import build_uav_prompt


def test_prompt_contains_state_and_progress():
    p = build_uav_prompt("Approach the red circle and stop 4.0 m away.", [0, 0, -10, 0], 1)
    assert p.startswith("In: What UAV action should be taken to")
    assert "State: [0, 0, -10, 0]." in p
    assert "Progress: 1." in p
    assert p.endswith("Out:")
