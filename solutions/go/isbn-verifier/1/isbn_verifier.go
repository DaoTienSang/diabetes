package isbnverifier

func IsValidISBN(isbn string) bool {
	s := ""
	for i := 0; i < len(isbn); i++ {
		if isbn[i] != '-' {
			s += string(isbn[i])
		}
	}

	if len(s) != 10 {
		return false
	}

	sum := 0

	for i := 0; i < len(s); i++ {
		digit := 0

		if s[i] == 'X' && i == 9 {
			digit = 10
		} else if s[i] >= '0' && s[i] <= '9' {
			digit = int(s[i] - '0')
		} else {
			return false
		}
		sum += digit * (10 - i)
	}

	return sum%11 == 0
}