from typing import Union, List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon


class Node:
    """ Class represnenting a node in the A* algorythm """

    def __init__(self, x_pos: int, y_pos: int):
        self.x = x_pos
        self.y = y_pos

        self.parent_node = None

        self.g_cost = 0
        self.h_cost = 0

    @property
    def F_cost(self) -> int:
        return self.g_cost + self.h_cost

    def __repr__(self):
        return f"{self.x, self.y}"

    def __lt__(self, other):
        if self.F_cost == other.F_cost:
            return self.g_cost < other.g_cost
        return self.F_cost < other.F_cost

    def __gt__(self, other):
        if self.F_cost == other.F_cost:
            return self.g_cost > other.g_cost
        return self.F_cost > other.F_cost

    def __hash__(self):
        return hash((self.x, self.y, self.h_cost, self.h_cost, self.F_cost, self.parent_node))


class GridNodes:
    """ Create a grid on which the A* algorithm takes place """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.grid = np.empty((width, height), dtype=object)
        for x in range(0, width):
            for y in range(0, height):
                self.grid[x][y] = Node(x, y)

    def get_node(self, x: int, y: int) -> Union[Node, None]:
        if x + 1 > self.width or y + 1 > self.height or x < 0 or y < 0:
            return None

        return self.grid[x][y]


class A_Star:
    EDGE_COLOR = "k"
    SQUARE_COLOR = "gray"
    BLOC_NODE = "k"
    OPENLIST_COLOR = "tab:blue"
    START_NODE = "yellow"
    END_NODE = "tab:green"
    CLOSED_COLOR = "tab:red"

    def __init__(self, grid_nodes: GridNodes, start_node, end_node):
        self.grid_nodes = grid_nodes
        self.start_node = start_node
        self.end_node = end_node

        self.bloc_nodes = []
        self.openList = []
        self.closedSet = set()

        self.width = self.grid_nodes.width
        self.height = self.grid_nodes.height
        self.iter_ = 0

        self.A_started = False
        self.selecting_square = True

        self._init_matplotlib_figure()

    def _init_matplotlib_figure(self):
        """ Initialize the matplotlib figure and the events """

        # Create figure and axes
        self.fig = plt.figure(figsize=((self.width + 2) / 3, (self.height + 2) / 3))
        self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9), aspect="equal", frameon=False,
                                    xlim=(-0.05, self.width + 0.05), ylim=(-0.05, self.height + 0.05))

        # Remove formatter
        for axis in (self.ax.xaxis, self.ax.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())

        # Create grid of squares
        self.squares = np.empty((self.width, self.height), dtype=object)
        for i in range(self.width):
            for j in range(self.height):
                self.squares[i][j] = RegularPolygon((i + 0.5, j + 0.5), numVertices=4, radius=0.5 * np.sqrt(2),
                                                    orientation=np.pi / 4, ec=self.EDGE_COLOR, fc=self.SQUARE_COLOR)

        # Add patches
        for sq in self.squares.flat:
            self.ax.add_patch(sq)

        # Add start node and end node
        self.squares[start_node.x, start_node.y].set_facecolor(self.START_NODE)
        self.squares[end_node.x, end_node.y].set_facecolor(self.END_NODE)

        # Create event hook for mouse clicks
        self.fig.canvas.mpl_connect("button_press_event", self._mouse_button_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._mouse_button_press)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Show
        plt.show()

    def _on_key_press(self, event):
        """ If event is spacebar """

        if event.key == " " and self.A_started is False:
            self.A_started = True
            self.run_A_star()

    def _mouse_button_press(self, event):
        """ On mouse left and right click event on matplotlib draw bloc nodes or remove them """

        if self.A_started or event.xdata is None or event.ydata is None:
            return

        i, j = int(event.xdata), int(event.ydata)

        # Left mouse button: draw bloc nodes
        if event.button == 1 and event.inaxes:
            self._click_square(i, j)

        # Right mouse button: remove blocs
        if event.button == 3 and event.inaxes:
            self._unclick_square(i, j)

        self.fig.canvas.draw()

    def _unclick_square(self, i, j):
        if self.end_node.x == i and self.end_node.y == j:
            return

        if self.start_node.x == i and self.start_node.y == j:
            return

        for node in self.bloc_nodes:
            if node.x == i and node.y == j:
                self.squares[i, j].set_facecolor(self.SQUARE_COLOR)
                self.bloc_nodes.remove(node)
                return

    def _click_square(self, i, j):
        if self.end_node.x == i and self.end_node.y == j:
            return

        if self.start_node.x == i and self.start_node.y == j:
            return

        self.squares[i, j].set_facecolor(self.BLOC_NODE)
        self.bloc_nodes.append(Node(i, j))

    def get_valid_nodes(self, current_node: Node) -> List[Node]:
        """ Retrieve a valid neighbor node """

        returned_nodes = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue

                eval_node = self.grid_nodes.get_node(current_node.x + i, current_node.y + j)

                if eval_node is None or eval_node in self.closedSet:
                    continue

                if any(node.x == eval_node.x and node.y == eval_node.y for node in self.bloc_nodes):
                    continue

                returned_nodes.append(eval_node)

        return returned_nodes

    def plot_path(self, current_node):
        """ Plot the path at each iteration """

        if current_node.x == start_node.x and current_node.y == start_node.y or \
                current_node.x == end_node.x and current_node.y == end_node.y:
            return

        for node in self.openList:
            if node.x == start_node.x and node.y == start_node.y or \
                    node.x == end_node.x and node.y == end_node.y:
                continue
            self.squares[node.x, node.y].set_facecolor(self.OPENLIST_COLOR)

        for node in self.closedSet:
            if node.x == start_node.x and node.y == start_node.y or \
                    node.x == end_node.x and node.y == end_node.y:
                continue
            self.squares[node.x, node.y].set_facecolor(self.CLOSED_COLOR)

        self.squares[current_node.x, current_node.y].set_facecolor(self.CLOSED_COLOR)

        self.ax.set_title("A* Algorithm Iteration: {}".format(self.iter_))
        plt.pause(0.001)
        self.fig.canvas.draw()

    def run_A_star(self) -> Node:
        """ Run the actual algorithm """

        self.openList.append(start_node)

        while len(self.openList) > 0:
            self.openList.sort()

            current_node = self.openList[0]
            self.openList.pop(0)

            self.plot_path(current_node)

            if (current_node.x, current_node.y) == (self.end_node.x, self.end_node.y):
                print("FOUND :D")
                return self.end_node

            # Add to closedSet
            self.closedSet.add(current_node)

            # Check valid neighbor node
            all_valid_nodes = self.get_valid_nodes(current_node)

            for valid_node in all_valid_nodes:
                if valid_node is None:
                    raise Exception("No valid node available, path not found.")

                # Calculate <g_cost> from start
                valid_node.g_cost = current_node.g_cost + get_distance(current_node, valid_node)


                # Calculate <h_cost> to end
                valid_node.h_cost = get_distance(valid_node, end_node)

                if valid_node not in self.openList:
                    valid_node.parent_node = current_node
                    self.openList.append(valid_node)

            self.iter_ += 1

        print("No valid path found snif :(.")


def get_distance(current_node: Node, valid_node: Node):
    """ Manhattan/cityblock distance """

    return abs(current_node.x - valid_node.x) + abs(current_node.y - valid_node.y)


if __name__ == "__main__":
    grid_nodes = GridNodes(15, 15)

    start_node: Node = Node(2, 2)
    end_node = Node(12, 12)

    A_Star(grid_nodes, start_node, end_node)

    plt.show()
