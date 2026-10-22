package proteintranslation

import "errors"

var ErrStop = errors.New("stop")
var ErrInvalidBase = errors.New("invalid codon")

func FromCodon(codon string) (string, error) {
    switch codon {
    case "AUG":
        return "Methionine", nil
    case "UUU", "UUC":
        return "Phenylalanine", nil
    case "UUA", "UUG":
        return "Leucine", nil
    case "UCU", "UCC", "UCA", "UCG":
        return "Serine", nil
    case "UAU", "UAC":
        return "Tyrosine", nil
    case "UGU", "UGC":
        return "Cysteine", nil
    case "UGG":
        return "Tryptophan", nil
    case "UAA", "UAG", "UGA":
        return "", ErrStop
    default:
        return "", ErrInvalidBase
    }
}

func FromRNA(rna string) ([]string, error) {
    result := []string{}

    for i := 0; i < len(rna); i += 3 {
        if i+3 > len(rna) {
            return nil, ErrInvalidBase
        }

        codon := rna[i : i+3]

        protein, err := FromCodon(codon)

        if err == ErrStop {
            break
        }

        if err != nil {
            return nil, err
        }

        result = append(result, protein)
    }

    return result, nil
}