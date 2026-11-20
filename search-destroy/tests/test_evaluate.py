import json

import numpy as np
from mcap.reader import make_reader

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.scripts.evaluate import fly, summarize


class Climb:
    def reset(self):
        self.calls = 0
        self.first_imu = None

    def act(self, observation):
        self.calls += 1
        assert "pose" not in observation
        if self.first_imu is None:
            self.first_imu = {key: observation[key].tolist() for key in ("gyro_rad_s", "accel_m_s2")}
        return np.array([0, 0, 0, .5])


def test_flight_is_logged_and_summarized(tmp_path):
    env = ApproachAviary(continuous=True, terrain="field@v1")
    agent = Climb()
    try:
        record = fly(env, agent, np.random.default_rng(0), tmp_path / "run.mcap", timeout_s=1., depth_every=5)
    finally:
        env.close()
    assert record["outcome"] == "timeout" and record["steps"] == 30
    with open(tmp_path / "run.mcap", "rb") as f:
        reader = make_reader(f)
        messages = [(channel.topic, message.data) for _, channel, message in reader.iter_messages()]
        topics = [topic for topic, _ in messages]
        assert reader.get_summary().statistics.metadata_count == 1
    assert topics.count("/agent/action") == 30 and topics.count("/truth") == 31
    assert topics.count("/observation/depth") == 7
    assert {"/observation/detection", "/observation/vio", "/imu"} <= set(topics)
    assert "/observation/pose" not in topics
    assert json.loads(next(data for topic, data in messages if topic == "/imu")) == agent.first_imu

    m = summarize([{"terrain": "field@v1", **record},
                   {"terrain": "field@v2", **record, "outcome": "success", "time_s": 10., "mean_speed_m_s": 4.}])["field"]
    assert m["success_rate"] == m["timeout_rate"] == .5
    assert m["reach_time_s"] == 10. and m["reach_mean_speed_m_s"] == 4.
    assert m["latency_mean_ms"] < 33
