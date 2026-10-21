package phonenumber

import "fmt"

func Number(phoneNumber string) (string, error) {
    s := ""
    hasPlus := false

    for i, c := range phoneNumber {
        if c >= '0' && c <= '9' {
            s += string(c)
        } else if c == '+' && i == 0 {
            hasPlus = true
        } else if c == ' ' || c == '-' || c == '.' ||
            c == '(' || c == ')' {
            continue
        } else {
            return "", fmt.Errorf("invalid character")
        }
    }


    if hasPlus && (len(s) != 11 || s[0] != '1') {
        return "", fmt.Errorf("invalid country code")
    }

    if len(s) == 11 {
        if s[0] != '1' {
            return "", fmt.Errorf("invalid country code")
        }
        s = s[1:]
    }

    if len(s) != 10 {
        return "", fmt.Errorf("invalid length")
    }


    if s[0] < '2' || s[0] > '9' ||
        s[3] < '2' || s[3] > '9' {
        return "", fmt.Errorf("invalid number")
    }

    return s, nil
}

func AreaCode(phoneNumber string) (string, error) {
    s, err := Number(phoneNumber)
    if err != nil {
        return "", err
    }

    return s[:3], nil
}

func Format(phoneNumber string) (string, error) {
    s, err := Number(phoneNumber)
    if err != nil {
        return "", err
    }

    return fmt.Sprintf("(%s) %s-%s", s[:3], s[3:6], s[6:]), nil
}