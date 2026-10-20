package luhn

func Valid(id string) bool {
    s := ""
    for i := 0; i < len(id); i++ {
        if id[i] != ' ' {
            if id[i] < '0' || id[i] > '9' {
                return false
            }
            s += string(id[i]) 
        }
    }

    if len(s) <= 1 {
        return false
    }
    sum := 0
    post := 1
	for i := len(s) - 1; i >= 0; i-- {
		digit := int(s[i] - '0') 

        if post % 2 == 0 {
        	digit *= 2   
            if digit > 9 {
                digit -= 9
            }
        }

        sum += digit
        post++ 
    }
    if sum % 10 == 0 {
        return true
    }
    return false
}
