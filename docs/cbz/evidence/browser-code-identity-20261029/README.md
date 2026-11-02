# Exact code-object identity repair

Parent: `8d24c37ff52c012294d71bbcbf08bd24fae4e22a`. A newly added regression first produced an actual assertion
failure: `fail.__code__.replace(co_filename='subject_fake.py')` is a distinct
but structurally equal code object and was accepted by equality-based membership.
The RED log is retained. Tuple storage plus explicit identity comparisons now
refuse the clone. A grader mutation restores equality and is caught by the new
regression, graded through the pristine structured origin classifier.

New observed population: 31 module /46 combined controls per mode; seven grader safeguard mutations.
Commands remain in the parent tool verification documentation; these tests do
not launch a browser or perform a full certificate replay. Pinned CPython3.11.16
is used for final local runs. Original production/workflow/packet bytes and all
prior test methods are preserved. Previous evidence is retained at its original
source, not relabeled as current. No full repository, hosted, mathematical or
CBZ-0 certification is claimed.
