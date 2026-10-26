package pov

type Tree struct {
	val      string
	children []*Tree
}

func New(value string, children ...*Tree) *Tree {
	return &Tree{
		val:      value,
		children: children,
	}
}

func (tr *Tree) Value() string {
	return tr.val
}

func (tr *Tree) Children() []*Tree {
	return tr.children
}

func (tr *Tree) String() string {
	if tr == nil {
		return "nil"
	}
	result := tr.Value()
	if len(tr.Children()) == 0 {
		return result
	}
	for _, ch := range tr.Children() {
		result += " " + ch.String()
	}
	return "(" + result + ")"
}

func findPath(root *Tree, target string) []*Tree {
	if root == nil {
		return nil
	}
	if root.val == target {
		return []*Tree{root}
	}
	for _, child := range root.children {
		p := findPath(child, target)
		if p != nil {
			return append([]*Tree{root}, p...)
		}
	}
	return nil
}

func cloneTree(tr *Tree) *Tree {
	if tr == nil {
		return nil
	}
	cloned := &Tree{val: tr.val}
	for _, ch := range tr.children {
		cloned.children = append(cloned.children, cloneTree(ch))
	}
	return cloned
}

func (tr *Tree) FromPov(from string) *Tree {
	path := findPath(tr, from)
	if path == nil {
		return nil
	}

	clonedPath := make([]*Tree, len(path))
	for i, node := range path {
		clonedPath[i] = &Tree{val: node.val}
		for _, ch := range node.children {
			if i+1 < len(path) && ch.val == path[i+1].val {
				continue
			}
			clonedPath[i].children = append(clonedPath[i].children, cloneTree(ch))
		}
	}

	for i := len(clonedPath) - 1; i > 0; i-- {
		child := clonedPath[i]
		parent := clonedPath[i-1]
		child.children = append(child.children, parent)
	}

	return clonedPath[len(clonedPath)-1]
}

func (tr *Tree) PathTo(from, to string) []string {
	reparented := tr.FromPov(from)
	if reparented == nil {
		return nil
	}

	targetPath := findPath(reparented, to)
	if targetPath == nil {
		return nil
	}

	res := make([]string, len(targetPath))
	for i, node := range targetPath {
		res[i] = node.val
	}
	return res
}