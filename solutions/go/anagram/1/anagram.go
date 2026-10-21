package anagram

import "strings"

func Detect(sub string, list []string) []string {
	var res []string
	src := []rune(strings.ToLower(sub))

	for _, c := range list {
		if strings.EqualFold(sub, c) {
			continue
		}

		dst := []rune(strings.ToLower(c))
		if len(src) != len(dst) {
			continue
		}

		used := make([]bool, len(dst))
		ok := true

		for _, r := range src {
			match := false
			for k, v := range dst {
				if r == v && !used[k] {
					used[k] = true
					match = true
					break
				}
			}
			if !match {
				ok = false
				break
			}
		}

		if ok {
			res = append(res, c)
		}
	}

	return res
}
