package clock

// Define the Clock type here.
import "fmt"
type Clock struct {
	m int 
}
func New(h, m int) Clock {
	total := h * 60 + m 
    total = ((total % 1440) + 1440) % 1440
    return Clock{m: total}
    
}

func (c Clock) Add(m int) Clock {
	return New(0, c.m + m)
} 

func (c Clock) Subtract(m int) Clock {
	return New(0, c.m - m)
}

func (c Clock) String() string {
	h := c.m / 60 
    m := c.m % 60 
    return fmt.Sprintf("%02d:%02d", h, m)
}
