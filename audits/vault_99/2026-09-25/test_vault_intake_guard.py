import copy, unittest
from vault_intake_guard import validate_entry
BASE={'drive_id':'X','title':'old','category':'A1','source_parent_id':'P','reason':'duplicate','authority_replacement_id':'Y','authority_replacement_title':'live','replacement_outside_vault':True,'sole_copy':False,'live_hold_research':False,'active_pkg_content':False,'routing_or_constitution':False,'manifest_row_prepared':True,'operator_confirmed':True}
class T(unittest.TestCase):
 def test_valid(self): self.assertTrue(validate_entry(BASE)['allowed'])
 def test_missing(self): x=copy.deepcopy(BASE);x.pop('drive_id');self.assertFalse(validate_entry(x)['allowed'])
 def test_no_replacement(self): x=copy.deepcopy(BASE);x['authority_replacement_id']='';self.assertFalse(validate_entry(x)['allowed'])
 def test_sole_copy(self): x=copy.deepcopy(BASE);x['sole_copy']=True;self.assertFalse(validate_entry(x)['allowed'])
 def test_hold(self): x=copy.deepcopy(BASE);x['live_hold_research']=True;self.assertFalse(validate_entry(x)['allowed'])
 def test_pkg(self): x=copy.deepcopy(BASE);x['active_pkg_content']=True;self.assertFalse(validate_entry(x)['allowed'])
 def test_routing(self): x=copy.deepcopy(BASE);x['routing_or_constitution']=True;self.assertFalse(validate_entry(x)['allowed'])
 def test_manifest(self): x=copy.deepcopy(BASE);x['manifest_row_prepared']=False;self.assertFalse(validate_entry(x)['allowed'])
 def test_confirmation(self): x=copy.deepcopy(BASE);x['operator_confirmed']=False;self.assertFalse(validate_entry(x)['allowed'])
 def test_bool_type(self): x=copy.deepcopy(BASE);x['sole_copy']=0;self.assertFalse(validate_entry(x)['allowed'])
 def test_category(self): x=copy.deepcopy(BASE);x['category']='A5';self.assertFalse(validate_entry(x)['allowed'])
if __name__=='__main__': unittest.main()
