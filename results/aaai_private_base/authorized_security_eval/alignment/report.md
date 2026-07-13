# Primary private-base proxy alignment diagnostic

Condition: `PRIMARY_PRIVATE_BASE_PROXY`  
View: `V2`  
Plaintext checkpoint supplied to attack: **no**

The diagnostic evaluated 12 real transformed-package cells: q/k/v/o projections
from layers 0, 12, and 23. The only comparison prior was independently randomized
same-shape S0 tensors. No trained unrelated same-architecture model was available,
so this is not yet the complete primary alignment matrix.

Across the 12 cells, spectral-profile cosine against the random control averaged
0.8497 (range 0.5585--0.9907), illustrating that coarse spectral similarity alone
is not evidence of basis recovery. Unpaired linear CKA averaged 0.2753. The
norm-sorted Procrustes relative error averaged 2.4348 (range 0.7292--18.4435), and
the row-norm OT relative distance averaged 2.3792. No plaintext basis, downstream
surrogate, or forward reconstruction was recovered in this primary diagnostic.

This result is partial: it does not establish failure of every unrelated-model
alignment method. A trained independently produced control, activation matching,
and downstream utility remain required.
