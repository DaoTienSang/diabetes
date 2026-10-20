package differenceofsquares

import "math"

func SquareOfSum(n int) int {
    tong := 0
    for i:= 1; i <= n; i++ {
        tong += i 
    }
	return int(math.Pow(float64(tong), 2))
}

func SumOfSquares(n int) int {
	tong := 0 
    for i:= 1; i <= n; i++ {
		tong += int(math.Pow(float64(i), 2))
    }
    return tong
}

func Difference(n int) int {
	return SquareOfSum(n) - SumOfSquares(n)
}
