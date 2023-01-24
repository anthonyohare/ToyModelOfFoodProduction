"""
A model of farm interactions where each farm grows food crops and through their management interacts with their neighbours.

This is the code for the paper:
O’Hare A (2023) A toy model of food production in a connected landscape. Front. Appl. Math. Stat. 9:1058273. doi: 10.3389/fams.2023.1058273


See also
http://science.tumblr.com/post/101026151217/quick-python-script-for-making-voronoi-polygons
"""
from math import hypot, exp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random



RADIUS = 250.0   # When generating the layout of farms use this radius
DPI=350

##################################################################################################
#    Define the farm class, containing their produce etc #########################################
#
##################################################################################################
##################################################################################################
class Farm:  # pylint: disable=too-few-public-methods
    """
    This class encapsulates a farm
    """

    def __init__(self, name, xcrd, ycrd, produce):
        """
        Create the farm object.
        :param id:   the id of the farm
        :param xcrd: the x/latitude coordinate of the farm.
        :param ycrd: the y/longitude coordinate of the farm.
        :param produce: the produce that the farm produces.
        """
        self.name = name
        self.xcrd = xcrd
        self.ycrd = ycrd
        self.produce = produce
        self.connected_farms = []  # any farms that are connected via roads, rivers etc.
        self.income = Income()  # the income (turnover, costs, etc) for the farm.

    def reset(self):
        self.income = Income()
        
    def __str__(self):
        return repr(self)

    def __repr__(self):
        # return ('Farm({0.name})').format(self)
        return (
            "Farm({0.name}, location=({0.xcrd:5.4f}, {0.ycrd:5.4f}), "
            "connected_farms=({connections}), "
            "produce={0.produce})\n"
        ).format(self, connections=[f.name for f in self.connected_farms])


##################################################################################################
#    Define the income class  ####################################################################
#
##################################################################################################
##################################################################################################
class Income:  # pylint: disable=too-few-public-methods
    """
    Class encapsulating the turnover and costs for a farm.
    """

    def __init__(self):
        """
        :param application_times:   the times that the farmer applies fertiliser/pestiside.
        :param initial_cost:   the initial set-up cost e.g. seed.
        :param regular_costs:  the regular cost of monitoring the crop
        """
        self.table = pd.DataFrame(columns=["time", "sales", "costs"])

    def add_cost(self, time, sales, cost):
        """
        Add a item for sales and a cost for a given time.
        :param time: the time of the cost/sales
        :param sales: the sales or value of the farms produce
        :param cost: the cost incurred on the farm in its production
        """
        last_sales = 0
        if self.table.count()[0] > 0:
            last_row = self.table.iloc[-1]
            last_sales = last_row[1]

        self.table  = pd.concat([self.table, 
            pd.DataFrame([[time, sales+last_sales, cost]],columns=["time", "sales", "costs"])], axis=0, ignore_index=True)


    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.table.get_string()


##################################################################################################
#    Define the growth models for each crop/produce type #########################################
#
##################################################################################################
##################################################################################################
class GrowthModel:  # pylint: disable=too-few-public-methods
    """
    Abstract class for growth models.
    """

    def value(self, *arg):
        """
        Calculate the value of the model for a given set of parameters.
        """
        pass


class LinearGrowthModel(GrowthModel):  # pylint: disable=too-few-public-methods
    """
    A linear growth model, the growth is a simple function of time
    """

    var1 = None
    var2 = None

    def __init__(self, *arg):
        """
        Initialise the parameters for the model.
        :param arg: the arguments for a linear growth model (1=coefficient for time,
                                                             2=constant [y-intercept])
        """
        self.var1 = arg[0]
        self.var2 = arg[1]

    def value(self, *arg):
        """
        Calculate the value of the model for a given set of parameters.
        :param arg: the arguments to the growth model (1=time)
        """
        return (self.var1 * arg[0]) + self.var2

    def __str__(self):
        return "linear growth model ({0.var1:5.4f}t+{0.var2:5.4f})\n".format(self)


class LogisticGrowthModel(GrowthModel):  # pylint: disable=too-few-public-methods
    """
    A sigmoidal growth model
    """

    max_value = None
    steepness = None
    midpoint = None

    def __init__(self, *arg):
        """
        Initialise the parameters for the model.
        :param arg: the arguments for a logistic growth model (1=maximum value of curve,
                                                               2=steepness,
                                                               3=mid/inflection point)
        """
        self.max_value = arg[0]
        self.steepness = arg[1]
        self.midpoint = arg[2]

    def value(self, *arg):
        """
        Calculate the value of the model for a given set of parameters.
        :param arg: the arguments to the growth model (1=time)
        """
        return self.max_value / (1 + exp(-self.steepness * (arg[0] - self.midpoint)))

    def __str__(self):
        return """logistic growth model ({0.max_value:5.4f}/
                  (1+e^(-{0.steepness:5.4f}t-{0.midpoint:5.4f})\n""".format(
            self
        )


##################################################################################################
#    Define the produce/crop classes #############################################################
#         and the respective costs
##################################################################################################
##################################################################################################
class ProduceType:  # pylint: disable=too-few-public-methods
    """
    Abstract class for types of produce.
    """

    pass


class RegularCosts:
    """
    Describes regular costs.
    """

    pass


class ProduceManagement:  # pylint: disable=too-few-public-methods
    """
    class containing the management configuration for each produce type
    """

    def __init__(
        self, application_times, initial_cost, regular_costs, 
        cost_per_application= 1.0, max_num_applications = 5
    ):
        """
        :param application_times:   a list of the times that the farmer applies fertiliser/pestiside.
        :param initial_cost:   the initial set-up cost e.g. seed.
        :param regular_costs:  the regular cost of monitoring the crop (a function of time).
        :param cost_per_application:  the cost of applying fertiliser/pestiside.
        :param max_num_applications:  the maximum number of aapplications that may be applied.
        """
        self.application_times = application_times
        self.initial_cost = initial_cost
        self.regular_costs = regular_costs
        self.cost_per_application = cost_per_application
        self.max_num_applications = max_num_applications

        # make sure that we have only used our management strategy up to the allowed number of times.
        assert len(application_times) <= max_num_applications

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            "Management("
            "initial_cost={0.initial_cost}, "
            "cost_per_application={0.cost_per_application}, max_num_applications={0.max_num_applications}, "
            "regular_costs={0.regular_costs}, "
            "application_times={0.application_times} )"
        ).format(self)


class UniformCosts(RegularCosts):
    """
    Uniform regular costs.
    """

    def __init__(self, cost):
        """
        :param cost: the cost incurred
        """
        self.cost = cost

    def amount(self, t):
        """
        Return the same cost at any time.
        :param t: the time the cost is incurred.
        """
        return self.cost

    def __repr__(self):
        return (
            "UniformCosts(cost={0.cost})"
        ).format(self)

class Produce1(ProduceType):  # pylint: disable=too-few-public-methods
    """
    Generic produce.
    """

    colour = "blue"
    marker = "*"
    desc = "TYPE1"
    linetype = "-"

    def __init__(self):
        self.manager = ProduceManagement([50, 100], 0.15, UniformCosts(0.001), cost_per_application=0.25, max_num_applications=5)
        #self.model = LinearGrowthModel(0.04, 0.02) # the model of how the produce grows.
        self.model = LogisticGrowthModel( 5.8, 0.7, 150)  # the model of how the produce grows.

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return ("Produce1(type={0.desc}, model={0.model}, " "{0.manager})").format(self)


class Produce2(ProduceType):  # pylint: disable=too-few-public-methods
    """
    Generic produce.
    """

    colour = "red"
    marker = "h"
    desc = "TYPE2"
    linetype = "--"

    def __init__(self):
        self.manager = ProduceManagement([10], 0.25, UniformCosts(0.001), cost_per_application=0.35, max_num_applications=10)
        self.model = LinearGrowthModel(0.028, 0.02) # the model of how the produce grows.
        #self.model = LogisticGrowthModel( 10.0, 0.8, 150)  # the model of how the produce grows.

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return ("Produce2(type={0.desc}, model={0.model}," "{0.manager})").format(self)


##################################################################################################
#    Create the interaction and distance matrix between farms  ###################################
#
##################################################################################################
##################################################################################################
def calc_farm_coupling_matrices(farms):
    """
    calculate the matrix of farm-farm distances and the interactions between farms
    (i.e. +1 if they produce the same produce and -1 otherwise).
    :param farms: the list of farms
    :return: tuple consisting of the farm-farm distance matrix and the farm-farm interaction matrix.
    """

    interaction_strength = 0.3
    positive_interaction = interaction_strength
    negative_interaction = -interaction_strength
    num_farms = len(farms)
    farm_distances = np.empty([num_farms, num_farms])
    farm_interactions = np.empty([num_farms, num_farms])
    farm_connections = np.zeros((num_farms, num_farms))
    for i in range(0, num_farms):
        farm_i = farms[i]
        for j in range(0, num_farms):
            farm_j = farms[j]

            if i == j:
                farm_distances[i][j] = 0.0
                farm_interactions[i][j] = interaction_strength
            else:
                farm_distances[i][j] = hypot( farm_i.xcrd - farm_j.xcrd, farm_i.ycrd - farm_j.ycrd)
                if farm_i.produce.desc == farm_j.produce.desc:
                    farm_interactions[i][j] = positive_interaction
                else:
                    farm_interactions[i][j] = negative_interaction

                if farm_j in farm_i.connected_farms:
                    farm_connections[i, j] = 1

    return (farm_distances, farm_interactions, farm_connections)


def linear_interaction_matrix(distances, interactions, connections, *args):
    """
    Create a matrix for the interactions between farms, each element of the matrix
    will be a function of time (since the intervention was applied). The assumptions are that:
    1. Each farmer applies the same intervention at the same time (t = equal for everyone)
    2. The value of the interaction is linear in both time and space
    :param distances: a matrix if distances between farms (must be square & symmetric).
    :param interactions: a matrix of 0,1s specifying if farms interact or not.
    :param args: any required arguments (1=coefficient for distance, 2=coefficient for time)
    :return: a matrix of functions, Note: the matrix will be returned as a dictionary where the
    keys are specified as tuples (i,j).
    """
    ret_matrix = {}
    for i in range(0, distances.shape[0]):
        for j in range(0, distances.shape[1]):

            def func(time, i=i, j=j):
                """
                Determine the interaction between farms i and j as a function of time.
                :param time: the time since the intervention was applied.
                :return: the interaction between farms i and j as a function of time.
                """
                assert isinstance(
                    time, (float, int, np.float64)
                ), "time argument for the interaction matrix must be singular"

                # if farms are connected, the distance between them is halved to model the 
                # the connection between them
                d = distances[i, j]
                if connections[i][j] == 1:
                    d *= 0.5

                # the interactions here are either positive or negative and the idea is that we add one to the sum
                # of these interactions to get a multiplicative factor
                interaction = 1.0
                if time >= 0:
                    interaction =  (interactions[i, j] * math.exp(-args[0] * d) * math.exp(-args[1] * time))
                    #print(f"{i},{j}  {interactions[i, j]}   {time}:{ math.exp(-args[1] * time)}         {d}:{math.exp(-args[0] * d)}    =    {interaction}")


                return interaction

            ret_matrix[i, j] = func
    return ret_matrix


##################################################################################################
#    Plot outputs etc  ###########################################################################
#
##################################################################################################
##################################################################################################
def plot_farm_locations(farms, descr):
    """
    Plot the locations of each farm, with a different colour and symbol for each farm
    type. Farms that are connected are joined by a line.
    :param farms: a list of farms to be plotted.
    :param descr: the prefix of the file where the plot will be saved.
    :return: nothing is returned from this method.
    """

    plt.figure(1, figsize=(4, 4))
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom="off",  # ticks along the bottom edge are off
        top="off",  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
        right="off",
        left="off",
        labelleft=False
    )
    for farm in farms:
        plt.scatter(
            farm.xcrd, farm.ycrd, color=farm.produce.colour, marker=farm.produce.marker
        )
        # join those farms that are connected.
        for other_farm in farm.connected_farms:
            print("connected")
            plt.plot( [farm.xcrd, other_farm.xcrd], [farm.ycrd, other_farm.ycrd], "k-", lw=2)

    plt.savefig(descr + "locations", dpi=DPI)

    # add farm id to the plot
    for farm in farms:
        # Sometimes we need to move the location of the farm id....
        ydelta = 4
        if farm.name == 17:
            ydelta = -6
        #if farm.produce.desc == "TYPE1":
        #    ydelta = -0.005
        xdelta = 2
        if farm.name == 10:
            xdelta = -10
        if farm.name == 17:
            xdelta = 3
        #if farm.name > 9:
        #    xdelta = 110

        plt.text(farm.xcrd+xdelta, farm.ycrd+ydelta, farm.name, fontsize=6)
    plt.savefig(descr + "locations_annotated", dpi=DPI)

    plt.close()

    import scipy
    try:
        points = np.array([[f.xcrd, f.ycrd] for f in farms])
        from scipy.spatial import ( Voronoi, voronoi_plot_2d,)  # pylint: disable=no-name-in-module

        vor = Voronoi(points)
        # Plot it:
        voronoi_plot_2d(vor)
        plt.savefig(descr + "veroni", dpi=DPI)
        plt.close()
    except scipy.spatial.qhull.QhullError:  # pylint: disable=no-member
        print("Could not calculate Veroni cells.")


def get_text_positions(x_data, y_data, text_data, plt):
    """
    Calculate the positions where the farm id and profit should be placed
    on the right hand side of the profit plots, rearranging them so that there
    is no overlappting text.
    :param x_data: a list of the x coordinates of the last point in the profit plot.
    :param y_data: a list of the y coordinates of the last point in the profit plot.
    :param text_data: a list of the text to be printed.
    :param plt: the plot object on which the text is to be printed.
    :return: nothing is returned from this method.
    """
    axs = plt.gca()
    fig = axs.get_figure()
    renderer = fig.canvas.get_renderer()

    # Create a list of Text objects, sorted to the order they appear in the plot (highest
    # profit first) and then calculate their bounding boxes.
    texts = [
        fig.text(x+25, y, z, fontsize=7) for x, y, z in zip(x_data, y_data, text_data)
    ]
    texts.sort(key=lambda x: x.get_position()[1], reverse=True)
    bboxes = [
        t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for t in texts
    ]

    # Determine the heights and ranges of the bounding boxes to give an idea of where to
    # plot the text.
    num = len(bboxes)
    x_max = max(x_data)+1
    y_max = max([b.ymax for b in bboxes]) * 1.2
    y_min = min([b.ymin for b in bboxes]) * 0.2
    y_height = (y_max - y_min) / num  # max([b.height for b in bboxes])*1.1
    centre = (y_max + y_min) / 2.0

    # make sure we have enough space
    top = max(y_max, math.ceil(num / 2) * y_height + y_height * 0.5)

    # update the positions of the text in each Text object and draw a line from the old
    # to the new position.
    for i, text in enumerate(texts):
        text_x, text_y = text.get_position()
        text.set_position((text_x, top))
        print(text_x, text_y, top, x_max)
        # draw a line from the original position to the new one
        plt.plot([x_max, text_x], [text_y, top], linestyle="-", color="k", linewidth=0.4)

        top = top - y_height

    return texts


def plot_farm_income(farms, descr):
    """
    Plot the sales, costs and profit for each farm over time.
    :param farms: a list of farms to be plotted.
    :param descr: the prefix of the file where the plot will be saved.
    :return: nothing is returned from this method.
    """

    cmap = plt.cm.get_cmap("rainbow", len(farms))

    # this is really bad but I can't successfully reuse the figures otherwise.
    def plot_sales():
        x = []
        y = []
        a = []
        for farm in farms:
            sales = list(farm.income.table["sales"])
            times = list(farm.income.table["time"])
            plt.figure(1)
            plt.xlabel("time (days)")
            plt.plot(times, sales, linestyle=farm.produce.linetype, c=farm.produce.colour)
            #plt.plot(times, sales, linestyle=farm.produce.linetype, c=cmap(farm.name))
        plt.savefig(descr + "sales", dpi=DPI)
        plt.close()

    def plot_costs():
        x = []
        y = []
        a = []
        for farm in farms:
            costs = list(farm.income.table["costs"])
            times = list(farm.income.table["time"])
            plt.figure(1)
            plt.xlabel("time (days)")
            plt.plot(times, costs, linestyle=farm.produce.linetype, c=farm.produce.colour)
            #plt.plot(times, costs, linestyle=farm.produce.linetype, c=cmap(farm.name))
        plt.savefig(descr + "costs", dpi=DPI)
        plt.close()

    def plot_profit():
        x = []
        y = []
        a = []
        for farm in farms:
            costs = list(farm.income.table["costs"])
            sales = list(farm.income.table["sales"])
            times = list(farm.income.table["time"])
            profit = [x1 - x2 for (x1, x2) in zip(sales, costs)]
            plt.figure(1)
            plt.xlabel("time (days)")
            plt.plot(times, profit, linestyle=farm.produce.linetype, c=farm.produce.colour)
            #plt.plot(times, profit, linestyle=farm.produce.linetype, c=cmap(farm.name))
        plt.savefig(descr + "profit", dpi=DPI)
        plt.close()

    def plot_profit_annotated():
        x = []
        y = []
        a = []
        for farm in farms:
            costs = list(farm.income.table["costs"])
            sales = list(farm.income.table["sales"])
            times = list(farm.income.table["time"])
            profit = [x1 - x2 for (x1, x2) in zip(sales, costs)]
            plt.figure(1)
            plt.xlabel("time (days)")

            x.append(times[-1])
            y.append(profit[-1])
            a.append(str(farm.name) + " (" + str(round(profit[-1], 4)) + ")")
            plt.plot(times, profit, linestyle=farm.produce.linetype, c=farm.produce.colour)
            #plt.plot(times, profit, linestyle=farm.produce.linetype, c=cmap(farm.name))

        plt.tight_layout()
        for text in get_text_positions(x, y, a, plt):
            x, y = text.get_position()
            plt.text(x, y, text.get_text(), rotation=0, color="black", fontsize=5)
        plt.tight_layout() 
        plt.savefig(descr + "profit_annotated", dpi=DPI)
        plt.close()

    plot_sales()
    plot_costs()
    plot_profit()
    plot_profit_annotated()


##################################################################################################
#    Initialise the system of farms, setting up their locations etc  #############################
#
##################################################################################################
##################################################################################################
def linear_homogeneous_system(num_farms):
    """
    Create a system of homogeneous farms laid out in a simple linear arrangement.
    :param num_farms: the number of farms to create.
    :return: a tuple of the list of farms and the layout description
    """
    return (
        [Farm(i, RADIUS*i / num_farms, 0.0, Produce1()) for i in range(num_farms)],
        "linear_homogeneous_",
    )


def circular_homogeneous_system(num_farms):
    """
    Create a system of homogeneous farms laid out in a simple circular arrangement.
    :param num_farms: the number of farms to create.
    :return: a tuple of the list of farms and the layout description
    """
    return (
        [
            Farm(
                i,
                RADIUS*np.cos(2 * np.pi * i / num_farms),
                RADIUS*np.sin(2 * np.pi * i / num_farms),
                Produce2(),
            )
            for i in range(num_farms)
        ],
        "circular_homogeneous_",
    )


def random_homogeneous_system(num_farms):
    """
    Create a system of homogeneous farms laid out randomly.
    :param num_farms: the number of farms to create.
    :return: a tuple of the list of farms and the layout description
    """
    return (
        [
            Farm(i, np.random.uniform(-RADIUS, RADIUS), np.random.uniform(-RADIUS, RADIUS), Produce1())
            for i in range(num_farms)
        ],
        "random_homogeneous_",
    )


def linear_mixed_system(num_farms, categories):
    """
    Create a system of farms (producing different output) laid out in a simple
    linear arrangement. We create [roughly] equal numbers of each category of
    farm distributed randomly.
    :param num_farms: the number of farms to create.
    :param categories: a list of the different produce that farms can grow.
    :return: a tuple of the list of farms and the layout description
    """
    """
    return (
        [
            Farm(i, RADIUS*i, 0.0, copy.deepcopy(np.random.choice(categories)))
            for i in range(num_farms)
        ],
        "linear_mixed_",
    )
    """
    # For the paper I used these.....
    return (
        [
            Farm(0, 0.0, 0.0, categories[1]),
            Farm(1, 250.0, 0.0, categories[1]),
            Farm(2, 500.0, 0.0, categories[1]),
            Farm(3, 750.0, 0.0, categories[0]),
            Farm(4, 1000.0, 0.0, categories[0]),
            Farm(5, 1250.0, 0.0, categories[1]),
            Farm(6, 1500.0, 0.0, categories[0]),
            Farm(7, 1750.0, 0.0, categories[1]),
            Farm(8, 2000.0, 0.0, categories[0]),
            Farm(9, 2250.0, 0.0, categories[0]),
            Farm(10, 2500.0, 0.0, categories[1]),
            Farm(11, 2750.0, 0.0, categories[1]),
            Farm(12, 3000.0, 0.0, categories[0]),
            Farm(13, 3250.0, 0.0, categories[0]),
            Farm(14, 3500.0, 0.0, categories[1]),
            Farm(15, 3750.0, 0.0, categories[0]),
            Farm(16, 4000.0, 0.0, categories[0]),
            Farm(17, 4250.0, 0.0, categories[0]),
            Farm(18, 4500.0, 0.0, categories[1]),
            Farm(19, 4750.0, 0.0, categories[1]),
            Farm(20, 5000.0, 0.0, categories[0]),
            Farm(21, 5250.0, 0.0, categories[0]),
            Farm(22, 5500.0, 0.0, categories[1]),
            Farm(23, 5750.0, 0.0, categories[1]),
            Farm(24, 6000.0, 0.0, categories[1]),            ],
        "linear_mixed_",
    )



def circular_mixed_random_system(num_farms, categories):
    """
    Create a system of farms (producing different output) laid out in a simple
    circular arrangement. We create [roughly] equal numbers of each category of
    farm distributed randomly.
    :param num_farms: the number of farms to create.
    :param categories: a list of the different produce that farms can grow.
    :return: a tuple of the list of farms and the layout description
    """
    """
    return (
        [
            Farm(
                i,
                RADIUS*np.cos(2 * np.pi * i / num_farms),
                RADIUS*np.sin(2 * np.pi * i / num_farms),
                copy.deepcopy(np.random.choice(categories)),
            )
            for i in range(num_farms)
        ],
        "circular_mixed_random_",
    )
        """
    # For the paper I used these.....
    return (
        [
            Farm(0, 250.0, 0.0, categories[1]),
            Farm(1, 242.14579028215778, 62.172471791213695, categories[0]),
            Farm(2, 219.0766700109659, 120.43841852542883, categories[0]),
            Farm(3, 182.24215685535287, 171.13677648217217, categories[0]),
            Farm(4, 133.95669874474916, 211.08198137550377, categories[0]),
            Farm(5, 77.25424859373686, 237.76412907378838, categories[1]),
            Farm(6, 15.697629882328378, 249.5066821070679, categories[1]),
            Farm(7, -46.84532864643115, 245.57181268217218, categories[0]),
            Farm(8, -106.44482289126819, 226.20676311650487, categories[0]),
            Farm(9, -159.35599743717242, 192.62831069394733, categories[1]),
            Farm(10, -202.25424859373683, 146.94631307311832, categories[1]),
            Farm(11, -232.44412147206285, 92.03113817116953, categories[1]),
            Farm(12, -248.02867532861944, 31.333308391076134, categories[1]),
            Farm(13, -248.02867532861947, -31.333308391076077, categories[0]),
            Farm(14, -232.44412147206288, -92.03113817116947, categories[1]),
            Farm(15,  -202.25424859373695, -146.94631307311818, categories[1]),
            Farm(16, -159.35599743717242, -192.62831069394733, categories[0]),
            Farm(17, -106.44482289126805, -226.20676311650496, categories[0]),
            Farm(18, -46.845328646431156, -245.57181268217218, categories[1]),
            Farm(19, 15.697629882328208, -249.5066821070679, categories[1]),
            Farm(20, 77.2542485937368, -237.7641290737884, categories[0]),
            Farm(21, 133.95669874474916, -211.08198137550377, categories[0]),
            Farm(22, 182.24215685535285, -171.13677648217222, categories[0]),
            Farm(23, 219.07667001096578, -120.43841852542903, categories[0]),
            Farm(24, 242.14579028215775, -62.17247179121384, categories[0]),            ],
        "circular_mixed_random_",
    )




def random_mixed_system(num_farms, categories):
    """
    Create a system of farms (producing different output) laid out randomnly.
    We create [roughly] equal numbers of each category of farm distributed
    randomly.
    :param num_farms: the number of farms to create.
    :param categories: a list of the different produce that farms can grow.
    :return: a tuple of the list of farms and the layout description
    """
    return (
        [
            Farm(
                i,
                RADIUS*np.random.uniform(0, 1.0),
                RADIUS*np.random.uniform(0, 1.0),
                copy.deepcopy(np.random.choice(categories)),
            )
            for i in range(num_farms)
        ],
        "random_mixed_",
    )

def random_mixed_25(categories):
    """
    Create a system of farms (producing different output) laid out randomnly (but set)
    We create [roughly] equal numbers of each category of farm distributed
    randomly.
    :param num_farms: the number of farms to create.
    :param categories: a list of the different produce that farms can grow.
    :return: a tuple of the list of farms and the layout description
    """
    return (
        [
            Farm(0, 4.37926858886073, 82.92721720245503, categories[0]),
            Farm(1, 126.87202473308987, 3.105974341328366, categories[0]),
            Farm(2, 205.47489173856027, 46.80286160915784, categories[0]),
            Farm(3, 196.7982961231806, 24.823618721470485, categories[0]),
            Farm(4, 138.9765348801231, 171.47741099025396, categories[0]),
            Farm(5, 23.289543527016455, 91.30499615292106, categories[0]),
            Farm(6, 222.20733963700607, 247.06279778906026, categories[0]),
            Farm(7, 37.61437774302151, 26.283142229487826, categories[0]),
            Farm(8, 108.87035741000614, 136.22465128794306, categories[0]),
            Farm(9, 178.15174773136224, 226.55459733450255, categories[0]),
            Farm(10, 25.671810285750745, 183.71793514785105, categories[0]),
            Farm(11, 65.66567430228484, 128.2868075339796, categories[0]),
            Farm(12, 135.64018467892183, 194.7812986343229, categories[0]),
            Farm(13, 109.74980313254329, 94.77474326805604, categories[0]),
            Farm(14, 151.53190317327372, 130.840245302136, categories[0]),
            Farm(15, 123.92075863560697, 86.33554338673112, categories[0]),
            Farm(16, 68.2777289207093, 202.19863158183585, categories[0]),
            Farm(17, 27.75191937587898, 182.98629540169415, categories[0]),
            Farm(18, 197.5017691243216, 160.1814312044118, categories[0]),
            Farm(19, 110.20582868244041, 111.53158155115477, categories[0]),
            Farm(20, 140.6459291249717, 89.70968288295198, categories[0]),
            Farm(21, 212.61427502282652, 99.1795639212712, categories[0]),
            Farm(22, 92.13008868092528, 35.45426711742689, categories[0]),
            Farm(23, 223.70357398085204, 54.45941369703533, categories[0]),
            Farm(24, 98.64168754240634, 23.2934783598783, categories[0]),        ],
        "random_25_",
    )


def random_mixed_from_origin_system(num_farms):
    """
    Create a system of farms (producing different output) laid out randomnly.
    We create different categories of farm depending on their distance from the
    origin.
    :param num_farms: the number of farms to create.
    :return: a tuple of the list of farms and the layout description
    """
    farms = []
    for i in range(num_farms):
        x_coord = np.random.uniform(-RADIUS, RADIUS)
        y_coord = np.random.uniform(-RADIUS, RADIUS)
        if hypot(x_coord, y_coord) < 0.6:
            farms.append(Farm(i, x_coord, y_coord, Produce1()))
        else:
            farms.append(Farm(i, x_coord, y_coord, Produce2()))
    return (farms, "random_mixed_from_origin_")


def random_mixed_by_latitude_system(num_farms):
    """
    Create a system of farms (producing different output) laid out randomnly.
    We create different categories of farm depending on their y coordinate.
    :param num_farms: the number of farms to create.
    :return: a tuple of the list of farms and the layout description
    """
    farms = []
    for i in range(num_farms):
        x_coords = np.random.uniform(-RADIUS, RADIUS)
        y_coords = np.random.uniform(-RADIUS, RADIUS)
        if y_coords < 0.5:
            farms.append(Farm(i, x_coords, y_coords, Produce1()))
        else:
            farms.append(Farm(i, x_coords, y_coords, Produce2()))
    return (farms, "random_mixed_by_latitude_")


def read_farm_data(file_name):
    """
    Read the list of farms from a pickled file.
    :param file_name: the name of the file from where we will read and unpickle
    the list of farms.
    :return: the list of farms, a description consisting of the name of the file
    (without the extension)
    """

    import dill as pickle
    import os

    # TODO - read as json NOT pickle
    with open(file_name, "rb") as fle:
        farms = pickle.load(fle)
    return farms, os.path.splitext(file_name)[0]


def save_farms(farms, file_name):
    """
    Save the list of farms to a [pickled] file
    :param farms: the list of farms to be saved.
    :param file_name: the name of the file to which we will save the farms.
    """

    import dill as pickle
    
    with open(file_name, "wb") as fle:
        pickle.dump(farms, fle)
    


def set_up_system(num_farms, fraction_connected, add_connections=True):
    """
    Set up the system of farms, connected them as necessary before calculating the
    distance and interaction matrices.
    :param num_farms: the number of farms to create
    :param fraction_connected: the fraction of farms to connect
    :param add_connections: whether or not we add connections to farms
    :return: a tuple consisting of the list of farms, a description of the system, a
    matrix of the farm distances, a matrix of the farm interactions, a [boolean] matrix
    of those farms that are connected, the interaction matrix consisting of functions of
    time for the interactions between each farm.
    """

    categories = [Produce1(), Produce2()]
    # farms, descr = linear_homogeneous_system(num_farms)
    # farms, descr = linear_mixed_system(num_farms, categories)
    # farms, descr = circular_homogeneous_system(num_farms)
    # farms, descr = circular_mixed_random_system(num_farms, categories)
    # farms, descr = random_homogeneous_system(num_farms)
    # farms, descr = random_mixed_system(num_farms, categories)
    # farms, descr = random_mixed_from_origin_system(num_farms)
    # farms, descr = random_mixed_by_latitude_system(num_farms)
    farms, descr = random_mixed_25(categories)
    # farms, descr = read_farm_data('random_25_original.pickle')

    # save the list of farms to file (without any connected farms)
    save_farms(farms, descr + "original.pickle")

    # In case we want to save and edit the layout!
    for farm in farms:
        print(f"Farm({farm.name}, {farm.xcrd}, {farm.ycrd}, categories[0]),")
    


    # if we want to use the saved farm layout without adding connections we
    # can set the connectFarms argument to False
    if add_connections:
        # connect farms, the connections may be road, rivers etc that allow pollution etc
        # to spread betewwn farms.
        
        # This is the random conneections I used in the paper.
        connected_farms = [farms[0], farms[5], farms[13], farms[20], farms[21]]
        farms[0].connected_farms.append(farms[5])
        farms[5].connected_farms.append(farms[0])
        farms[5].connected_farms.append(farms[13])
        farms[13].connected_farms.append(farms[5])
        farms[13].connected_farms.append(farms[20])
        farms[20].connected_farms.append(farms[13])
        farms[20].connected_farms.append(farms[21])
        farms[21].connected_farms.append(farms[20])
        
        
        """
        # Randonly connect farms
        connected_farms = np.random.choice(
            farms, int(num_farms * fraction_connected), replace=False
        )
        for this in connected_farms:
            shortest_dist = 100.0  # larger than any possible distance in our simulation
            neighbour = None
            for other in connected_farms:
                distance = hypot(this.xcrd - other.xcrd, this.ycrd - other.ycrd)
                if (
                    other not in this.connected_farms
                    and this != other
                    and distance < shortest_dist
                ):
                    shortest_dist = distance
                    neighbour = other
            if neighbour is not None:
                this.connected_farms.append(neighbour)
                neighbour.connected_farms.append(this)
        """
        # save the list of farms to file (without any connected farms)
        save_farms(farms, descr + "with_new_connections.pickle")
        print("connected_farms")
        print (connected_farms)

    plot_farm_locations(farms, descr)

    # Set up the farm-farm interaction matrix for the problem.
    (farm_distances, farm_interactions, farm_connections) = calc_farm_coupling_matrices(farms)
    interaction_matrix = linear_interaction_matrix( farm_distances, farm_interactions, farm_connections, 0.07, 0.008)
    print("Distances")
    print (farm_distances)
    print("Interactions")
    print (farm_interactions)
    print("Connections")
    print (farm_connections)
    #print("interactions")
    #print (interaction_matrix)

    # Check everything we set up makes sense - e.g. interaction and distance matrices are
    # the same size etc.
    assert (
        farm_distances.shape[0] == farm_interactions.shape[0]
    ), "The farm distances and interactions matrices must have same dimensions (0)"
    assert (
        farm_distances.shape[1] == farm_interactions.shape[1]
    ), "The farm distances and interactions matrices must have same dimensions (1)"
    assert (
        farm_distances.shape[0] == farm_connections.shape[0]
    ), "The farm distances and connections matrices must have same dimensions (0)"
    assert (
        farm_distances.shape[1] == farm_connections.shape[1]
    ), "The farm distances and connections matrices must have same dimensions (1)"
    assert farm_distances.shape[1] == np.sqrt(
        len(interaction_matrix)
    ), "The interaction matrix must be square with same dimensions as farm data"

    return (
        farms,
        descr,
        farm_distances,
        farm_interactions,
        farm_connections,
        interaction_matrix,
    )


##################################################################################################
#    Run the simulation of a list of farms growing their produce under their given management  ###
#    strategy
##################################################################################################
##################################################################################################


def run_simulation(farms, interaction_matrix, max_time):
    """
    Run the simulation of the list of farms.
    :param farms: the list of farms
    :param interaction_matrix: todo explain
    :param max_time: maximum time for the simulation to run.
    """

    def calc_cost(farm, time):
        """
        Calculate the cost of food produced on a given farm at a given time. The cost is made up
        of a one time initial cost (e.g. seed), a regular cost e.g. for continuous monitoring of
        the crop, which we will integrate up to time t, and the cost of each strategy.
        :param farm: the farm where we will calculate the food production.
        :param time: the time at which the food production value is to be calculated.
        """
        total_cost = farm.produce.manager.initial_cost
        for t in range(0, time):
            total_cost += farm.produce.manager.regular_costs.amount(t)

        # Add the integrated costs up to and including the current time
        for index, value in enumerate(farm.produce.manager.application_times):
            if value <= time:
                total_cost += farm.produce.manager.cost_per_application

        return total_cost

    def production_value(farm, time):
        """
        Calculate the amount of food produced on a given farm at a given time. Integrate the sum
        of all interactions of this farm with their neighbours given a vector of the times
        everyone made some intervention.
        :param farm: the farm where we will calculate the food production.
        :param time: the time at which the food production value is to be calculated.
        """
        total_interaction = 1.0
        farm_index = farms.index(farm)
        for other_farm in farms:
            other_farm_index = farms.index(other_farm)
            for application in other_farm.produce.manager.application_times:
                if time > application:
                    total_interaction += interaction_matrix[farm_index, other_farm_index](time - application)
                    #print (f"{time} {farm.name}-{farm.produce.desc},  {other_farm.name}-{other_farm.produce.desc}, {hypot( farm.xcrd - other_farm.xcrd, farm.ycrd - other_farm.ycrd)}     {application},{time-application}          {interaction_matrix[farm_index, other_farm_index](time - application)}  {total_interaction}")

        return (farm.produce.model.value(time)- farm.produce.model.value(time-1)) * (1+total_interaction)

    # TODO: Suppose we pick one farm at random and try to determine their extra [application] cost to
    # maintain their output.
    # trial_farm = farms[14]
    # trial_farm.produce.manager.application_times.append(53)

    # TODO: need to link the regular cost and/or application to the growth function - they
    # are linked here via a scaling factor determined by the interaction matrix. Make sure that
    # it increases the values locally.
    for time in range(0, max_time):
        for farm in farms:
            farm.income.add_cost(time, production_value(farm, time), calc_cost(farm, time))




def calcPollution(farms, max_time, gamma, phi):


    def plot_pollution(matrix, produce):
        from matplotlib import pyplot as plt
        plt.figure(1, figsize=(4, 4))
        plt.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom="off",  # ticks along the bottom edge are off
            top="off",  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            right="off",
            left="off",
            labelleft=False
        )
        im = plt.imshow(matrix)
        cbar = plt.colorbar(im, location='right', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"pollution_{produce}.png")
        plt.close()

    size = 500
    interaction_strength = 0.3
    grid_size = RADIUS / size

    # Define a matrix
    pollution_1 = np.empty([size, size])
    pollution_2 = np.empty([size, size])

    for i in range(0, size):
        for j in range(0, size):

            # get the coordinates of the centre if the mesh points.
            mesh_x = i * grid_size
            mesh_y = j * grid_size

            interaction_1 = 1.0
            interaction_2 = 1.0
            for farm in farms:
                for application in farm.produce.manager.application_times:
                    if max_time > application:
                        d =  hypot( farm.xcrd - mesh_x, farm.ycrd - mesh_y)
                        time = max_time - application
                        if farm.produce.desc == "TYPE1":
                            interaction_1 += (interaction_strength * math.exp(-gamma * d) * math.exp(-phi * time))
                        elif farm.produce.desc == "TYPE2":
                            interaction_2 += (interaction_strength * math.exp(-gamma * d) * math.exp(-phi * time))
                pollution_1[i][j] = interaction_1
                pollution_2[i][j] = interaction_2

    plot_pollution(np.rot90(pollution_1), "star")
    plot_pollution(np.rot90(pollution_2), "hexagon")

    return (pollution_1, pollution_2)




 

##################################################################################################
##################################################################################################
##################################################################################################

if __name__ == "__main__":
    print("Starting")

    NUM_FARMS = 25
    FRACTION_CONNECTED = 0.2
    MAX_TIME = 240
    ADD_NEW_CONNECTIONS = (
        False
    )  # add connections to the farms we create, if we read in a saved
    # set then this will determine whether or not we add new connections.

    FARMS, DESCR, FARM_DISTANCES, FARM_INTERACTIONS, FARM_CONNECTIONS, INTERACTION_MATRIX = set_up_system(
        NUM_FARMS, FRACTION_CONNECTED, ADD_NEW_CONNECTIONS
    )

    # Run the simulation.
    run_simulation(FARMS, INTERACTION_MATRIX, MAX_TIME)
    plot_farm_income(FARMS, DESCR)

    # Print a summary of the simulation
    for farm in FARMS:
        costs = list(farm.income.table["costs"])[-1]
        sales = list(farm.income.table["sales"])[-1]
        profit = sales - costs

        print(f"{farm.name}  {farm.produce.desc}    {farm.xcrd}  {farm.ycrd}  {profit}")


    # Now try and calculate the pollution in the system, i.e. how much of the 'interactions' are
    # felt in the landscape. First define a mesh over the landscape and calculate the interactions at 
    # at each mesh point with and without any ProduceManagement on each farm (if there are no
    # ProduceManagement there there will be no interactions!)

    #(pollution_1, pollution_2) = calcPollution(FARMS, MAX_TIME, 0.07, 0.008)



    # TODO - need to optimise the farm.produce.produceManagement.applicationTimes for maximum profit
    # Note each application time has an associated farm.produce.produceManagement.applicationCost
    # and we should put a constraint on the times so that the costs < some limit

    """
    farms = FARMS.copy()
    best_farms = farms
    for n in range(0, 3):
        prev_benefit = 0.0

        # 1. Randomly pick the days when the applications are applied on each farm.
        for farm in farms:
            appls =  np.random.randint(0, farm.produce.manager.max_num_applications+1)
            farm.produce.manager.application_times = random.sample(range(0, MAX_TIME), appls)
            farm.reset()

        # 2. reset the farms and run the simulation
        run_simulation(farms, INTERACTION_MATRIX, MAX_TIME)

        # 3. Calculate the benefit/payoff
        benefit = 0.0
        for farm in farms:
            costs = list(farm.income.table["costs"])
            sales = list(farm.income.table["sales"])
            profit = [x1 - x2 for (x1, x2) in zip(sales, costs)]
            benefit += sum(profit)

        # 4. if the benefit is greater than the previous one, accept the farms as the better, 
        #    else accept with a coin toss.
        if benefit >= prev_benefit or (prev_benefit>benefit and np.random.random() > 0.3):
            prev_benefit = benefit
            best_farms = farms
        print(benefit)
        plot_farm_income(best_farms, DESCR)
    """



# vor.points	(ndarray of double, shape (npoints, ndim)) Coordinates of input points.
# vor.vertices	(ndarray of double, shape (nvertices, ndim)) Coordinates of the Voronoi vertices.
# vor.ridge_points	(ndarray of ints, shape (nridges, 2)) Indices of the points between which
#                    each Voronoi ridge lies.
# vor.ridge_vertices	(list of list of ints, shape (nridges, *)) Indices of the Voronoi vertices
#                    forming each Voronoi ridge.
# vor.regions	(list of list of ints, shape (nregions, *)) Indices of the Voronoi vertices forming
#                    each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
# vor.point_region	(list of ints, shape (npoints)) Index of the Voronoi region for each input
#                    point. If qhull option “Qc” was not specified, the list will contain -1 for
#                    points that are not associated with a Voronoi region.
