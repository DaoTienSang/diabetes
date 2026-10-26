package linkedlist

import "errors"

var ErrEmptyList = errors.New("list is empty")

type Node struct {
	Value any
	next  *Node
	prev  *Node
}

func (n *Node) Next() *Node {
	return n.next
}

func (n *Node) Prev() *Node {
	return n.prev
}

type List struct {
	head *Node
	tail *Node
	size int
}

func NewList(elements ...any) *List {
	l := &List{}
	for _, elem := range elements {
		l.Push(elem)
	}
	return l
}

func (l *List) First() *Node {
	return l.head
}

func (l *List) Last() *Node {
	return l.tail
}

func (l *List) Count() int {
	return l.size
}

func (l *List) Push(v any) {
	node := &Node{Value: v}
	if l.tail == nil {
		l.head = node
		l.tail = node
	} else {
		node.prev = l.tail
		l.tail.next = node
		l.tail = node
	}
	l.size++
}

func (l *List) Unshift(v any) {
	node := &Node{Value: v}
	if l.head == nil {
		l.head = node
		l.tail = node
	} else {
		node.next = l.head
		l.head.prev = node
		l.head = node
	}
	l.size++
}

func (l *List) Pop() (any, error) {
	if l.tail == nil {
		return nil, ErrEmptyList
	}
	val := l.tail.Value
	if l.head == l.tail {
		l.head = nil
		l.tail = nil
	} else {
		l.tail = l.tail.prev
		l.tail.next = nil
	}
	l.size--
	return val, nil
}

func (l *List) Shift() (any, error) {
	if l.head == nil {
		return nil, ErrEmptyList
	}
	val := l.head.Value
	if l.head == l.tail {
		l.head = nil
		l.tail = nil
	} else {
		l.head = l.head.next
		l.head.prev = nil
	}
	l.size--
	return val, nil
}

func (l *List) Reverse() {
	curr := l.head
	l.tail = l.head
	var prev *Node
	for curr != nil {
		curr.prev = curr.next
		curr.next = prev
		prev = curr
		curr = curr.prev
	}
	l.head = prev
}

func (ll *List) Delete(v any) bool {
	curr := ll.head
	for curr != nil {
		if curr.Value == v {
			if curr == ll.head && curr == ll.tail {
				ll.head = nil
				ll.tail = nil
			} else if curr == ll.head {
				ll.head = curr.next
				ll.head.prev = nil
			} else if curr == ll.tail {
				ll.tail = curr.prev
				ll.tail.next = nil
			} else {
				curr.prev.next = curr.next
				curr.next.prev = curr.prev
			}
			ll.size--
			return true
		}
		curr = curr.next
	}
	return false
}