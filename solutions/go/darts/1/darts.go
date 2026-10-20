package darts

import "math"

func Score(x, y float64) int {
	kc := math.Sqrt(math.Pow(x, 2) + math.Pow(y, 2)) 
    if kc <= 1 {
        return 10 
    } else if kc <= 5 {
        return 5
    } else if kc <= 10 {
        return 1
    }
    return 0 
}
