#!/bin/bash
INIT_CONTRAST=CONTRAST_1.2

for contrast in 1.5 1.8; do
    NEXT_CONTRAST=CONTRAST_${contrast}
    echo ${INIT_CONTRAST} ${NEXT_CONTRAST}
    python surf_lbp.py && sed -i 's/'${INIT_CONTRAST}'/'${INIT_CONTRAST}'/g' surf_lbp.txt
    INIT_CONTRAST=CONTRAST_${contrast}
done