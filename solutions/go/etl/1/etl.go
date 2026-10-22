package etl

import "strings"

func Transform(input map[int][]string) map[string]int {
    result := make(map[string]int)

    for score, letters := range input {
        for i := 0; i < len(letters); i++ {
            letter := strings.ToLower(letters[i])
            result[letter] = score
        }
    }

    return result
}