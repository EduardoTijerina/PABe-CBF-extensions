from pabe_cbf.scenarios import Participant, Scenario
from pabe_cbf.g1.geometry import G1_GEOM
from pabe_cbf.g1.episode import G1Episode
from pabe_cbf.core import METHODS

sc = Scenario("iii_withdraw_contact", Participant(5, geom=G1_GEOM), 0)
res, _, _ = G1Episode(METHODS["B1"], sc, None).run()
print(res["R_reason"], res["peak_F"], res["V_any"])
