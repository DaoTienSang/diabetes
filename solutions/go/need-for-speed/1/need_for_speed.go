package speed

// Car represents a remote controlled car.
type Car struct {
    battery      int
    batteryDrain int
    speed        int
    distance     int
}

// NewCar creates a new car.
func NewCar(speed, batteryDrain int) Car {
    return Car{
        battery:      100,
        batteryDrain: batteryDrain,
        speed:        speed,
        distance:     0,
    }
}

// Track represents a race track.
type Track struct {
    distance int
}

// NewTrack creates a new track.
func NewTrack(distance int) Track {
    return Track{distance: distance}
}

// Drive drives the car if it has enough battery.
func Drive(car Car) Car {
    if car.battery >= car.batteryDrain {
        car.distance += car.speed
        car.battery -= car.batteryDrain
    }
    return car
}

// CanFinish checks if the car can finish the track.
func CanFinish(car Car, track Track) bool {
    if track.distance <= 0 {
        return true
    }

    if car.speed <= 0 {
        return false
    }

    if car.batteryDrain == 0 {
        return true
    }

    maxDrives := car.battery / car.batteryDrain
    return maxDrives * car.speed >= track.distance
}