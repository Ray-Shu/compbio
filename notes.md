# Notes

### 4-track Encodings

A one-hot encoding of the RNA sequence. So if I have a sequence of length d, then my input is a matrix of size (4,d).

### 6-track Encodings

Includes the one-hot encoding of the RNA sequence like the 4 track but also adds two more pieces of information.

5'th row (5'th track) are the splice site indicators. A gene is broken into *exons* and *introns*. We only care about exons, so we splice out introns and stitch the exons together. 

6'th row are the codon position markers. Marks the first nucleotide of each codon. 

---

### saRNA Limitations

saRNA is already pre-spliced and doesn't have information on the spliced exons. i.e, there is no information given to track 5. This leaves us with two options:

1. Pad track 5 with zeroes to use the 6-track model and still have track 6 for extra info
2. Use the 4 track model with only one-hots of the nucleotide sequence.

---
### Possible Plans 

#### Plan 1

Try using the data from the "Next-Generation saRNA Platform". 

The excel sheet gives the alpharivuses and their access number so we have the full strand. We can perform linear probing (ie. freeze model weights and add a single linear layer to do downstream tasks). 

Fine-tuning with this low data likely overfits but may be worth trying? 


#### Plan 2

abc

---
### Metric Names
1. Immune response: RIG-I, MDA5, IFIT fold changes (want < 1)
2. Replication rate: GFP fold change ÷ nsP3 fold change (want high ratio)
3. Protein expression: GFP fold change alone (want > 1)

---
### Terminology 

ORF: open reading frame. The sequence that gets read and translated into a protein. Has a start codon (AUG) and stop codon (UAA, UAG, or UGA). Everything in-between is what the ribosomes read.

UTR: untranslated region. The parts around the ORF that tell the ribosomes where and how to land and start reading.

- In saRNA, the 5' and 3' contain **conserved sequence elements** that the replicase recognizes to initiate self-amlification. 