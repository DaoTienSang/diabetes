package listops

type IntList []int

func (s IntList) Foldl(fn func(int, int) int, initial int) int {
    acc := initial

    for i := 0; i < len(s); i++ {
        acc = fn(acc, s[i])
    }

    return acc
}

func (s IntList) Foldr(fn func(int, int) int, initial int) int {
    acc := initial

    for i := len(s) - 1; i >= 0; i-- {
        acc = fn(s[i], acc)
    }

    return acc
}

func (s IntList) Filter(fn func(int) bool) IntList {
    result := make(IntList, len(s))
    count := 0

    for i := 0; i < len(s); i++ {
        if fn(s[i]) {
            result[count] = s[i]
            count++
        }
    }

    return result[:count]
}

func (s IntList) Length() int {
    count := 0

    for range s {
        count++
    }

    return count
}

func (s IntList) Map(fn func(int) int) IntList {
    result := make(IntList, len(s))

    for i := 0; i < len(s); i++ {
        result[i] = fn(s[i])
    }

    return result
}

func (s IntList) Reverse() IntList {
    result := make(IntList, len(s))

    for i := 0; i < len(s); i++ {
        result[i] = s[len(s)-1-i]
    }

    return result
}

func (s IntList) Append(lst IntList) IntList {
    result := make(IntList, len(s)+len(lst))

    for i := 0; i < len(s); i++ {
        result[i] = s[i]
    }

    for i := 0; i < len(lst); i++ {
        result[len(s)+i] = lst[i]
    }

    return result
}

func (s IntList) Concat(lists []IntList) IntList {
    result := make(IntList, len(s))

    for i := 0; i < len(s); i++ {
        result[i] = s[i]
    }

    for i := 0; i < len(lists); i++ {
        result = result.Append(lists[i])
    }

    return result
}