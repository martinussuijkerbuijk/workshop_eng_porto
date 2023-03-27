import bpy
import numpy as np
import sys, subprocess, os

# This installs the necessary dependencies first if they are missing
try:
    import supershape as sshape
    import pandas as pd

    print("Imported packages supershape and pandas")
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "supershape", "pandas"])

    import supershape as sshape
    import pandas as pd

    print("Missing modules installed!")

# Creates dataframe of the 6 variables of superformula function.
# Check for function implementation http://paulbourke.net/geometry/supershape/
# For a wonderful explanation by Johan Gielis himself check https://www.youtube.com/watch?v=N61nzcy6pKU
def generate_population(min_v=0.001, max_v=10, size=(25), shift=0, gen_nr=0):

    dict = {}

    m = np.random.uniform(1., 60., size)
    a = np.random.uniform(.1, 20., size)
    b = np.random.uniform(.1, 20., size)
    n1 = np.random.uniform(.01, 30., size)
    n2 = np.random.uniform(.01, 30., size)
    n3 = np.random.uniform(.01, 30., size)

    # Generate dictionary
    for i in range(size):
        dict[f'nr_{i + shift}'] = {'gen': gen_nr, 'm': m[i], 'a': a[i], 'b': b[i], 'n1': n1[i], 'n2': n2[i],
                                   'n3': n3[i]}

    # Convert to DataFrame for easy access later
    df = pd.DataFrame(dict).T

    return df

# Scaling function as some parameter combinations yield very large aspect ratios
def set_height(height):
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')

    # Store in variable
    objs = bpy.context.selected_objects
    for o in objs:
        try:
            factor = height / o.dimensions.y
        except ZeroDivisionError as arr:
            print('Cannot divide by zero!')
            factor = height / 0.1
        o.scale = (factor, factor, factor)
    return


def crossover_mutate(parents: pd.DataFrame, crossover=True, mutate=True, pop=25):
    """"
    inputs:
        parents[]

    outputs:
        Is Always set to 25 items

    When crossover is set, this will increase the diversity but leads to slower convergence
    """

    parents['gen'] += 1

    # Check if selection is not too large to maintain diversity
    if len(parents) > (pop // 2) == True:
        print("Please select less than half of the generation. \
                Otherwise there will be no diveristy")

    else:

        survivors = generate_population(size=(25 - len(parents)), shift=len(parents), gen_nr=parents['gen'][0])
        parents_survivors = [parents, survivors]
        new_gen = pd.concat(parents_survivors)

        cache_new_gen = new_gen.copy(deep=True)

        print(f"Length parents is: {len(parents)}")

        # Fill population with new mebers + crossover
        # We're not saving the parents, for the sake of diversity and since we have a save function
        if crossover == True:
            for i in range(len(parents)):
                try:

                    if i % 2 == 0:
                        # 50% change to swap mab or n1:n3
                        prob = np.random.randint(1, 10, 1)
                        if prob % 2 == 0:
                            new_gen.loc[f'nr_{i + len(parents)}']['m':'b'] = cache_new_gen.loc[f'nr_{i + 1}']['m':'b']
                        else:
                            new_gen.loc[f'nr_{i + len(parents)}']['n1':'n3'] = cache_new_gen.loc[f'nr_{i + 1}'][
                                                                               'n1':'n3']
                    else:
                        if prob % 2 == 0:
                            new_gen.loc[f'nr_{i + len(parents)}']['m':'b'] = cache_new_gen.loc[f'nr_{i - 1}']['m':'b']
                        else:
                            new_gen.loc[f'nr_{i + len(parents)}']['n1':'n3'] = cache_new_gen.loc[f'nr_{i - 1}'][
                                                                               'n1':'n3']

                            # if the length is uneven swap the lat one with a random genotype
                    if len(parents) % 2 != 0 and i == len(parents) - 1:
                        if prob % 2 == 0:
                            new_gen.loc[f'nr_{i + 1 + len(parents)}']['m':'b'] = cache_new_gen.loc[f'nr_{i}']['m':'b']
                        else:
                            new_gen.loc[f'nr_{i + 1 + len(parents)}']['n1':'n3'] = cache_new_gen.loc[f'nr_{i}'][
                                                                                   'n1':'n3']

                except KeyError as err:
                    print("Please don't select all object as this won't lead to convergence")

    # This also applies mutation
    if mutate == True:
        new_gen_mutate = mutate_generation(new_gen, parents)

    if mutate == False and crossover == False:
        print("Please set flag crossover or mutate")

    return new_gen_mutate


def mutate_generation(generation: pd.DataFrame, parents: pd.DataFrame):
    keys = ['m', 'a', 'b', 'n1', 'n2', 'n3']

    for i in range(len(parents)):
        tolerance = np.random.uniform(-0.5, 0.5, size=(1))
        key = np.random.choice(keys)
        # add some probability for mutation
        prob = np.random.randint(1, 100, 1)
        if prob <= 33:
            generation.loc[f'nr_{i}'][key] += tolerance
            print(f"Added mutation to phenotype nr_{i} with key {key}")

    return generation


def generate_ss_from_file(path: str, step_x=3., shape=(20, 20)):
    # First check if collection exists, if not create collection for objects
    if bpy.context.scene.collection.children.get("ShapeCollection"):
        collection = bpy.context.scene.collection.children.get("ShapeCollection")
    else:
        collection = bpy.data.collections.new("ShapeCollection")
        bpy.context.scene.collection.children.link(collection)

    df = pd.read_pickle(path)
    matrix = df.to_numpy()
    matrix = matrix[:, 1:]

    for values in matrix:
        obj = sshape.make_bpy_mesh(shape, name=f'supershape', weld=True)

        x, y, z = sshape.supercoords(
            # m, a, b, n1, n2, n3 (1x6 or 2x6)
            values,
            # u,v resolution
            shape=(shape)
        )
        sshape.update_bpy_mesh(x, y, z, obj)

        # move object to new collection
        if obj.users_collection[0] == collection:
            continue
        else:
            collection.objects.link(obj)



def parent_selection(objs, save=True):
    # get data from selected objects
    names = [o.name for o in objs]

    # Save selected items to separate pickle object
    if save == True:
        df = pd.read_pickle(bpy.path.abspath("//" + "dataframe.pkl"))
        parents = df.loc[names]
        parents.index = [f'nr_{i}' for i in range(len(names))]

        # Check if file exists, it adds all new evolutions to one file. Delete saved_selection.pkl if you want to start fresh
        if os.path.exists(bpy.path.abspath("//" + "saved_selection.pkl")):

            prev_generations = pd.read_pickle(bpy.path.abspath("//" + "saved_selection.pkl"))
            combine = [parents, prev_generations]
            all_saved = pd.concat(combine)
            all_saved.index = [f'nr_{i}' for i in range(len(all_saved))]

            all_saved.to_pickle(bpy.path.abspath("//" + "saved_selection.pkl"))
            print(f"Saved selection to file {bpy.path.abspath('//')}saved_selection.pkl")

        else:
            # Just Save the last generation
            parents.to_pickle(bpy.path.abspath("//" + "saved_selection.pkl"))


    else:
        # Read pickle
        df = pd.read_pickle(bpy.path.abspath("//" + "dataframe.pkl"))
        parents = df.loc[names]
        parents.index = [f'nr_{i}' for i in range(len(names))]
        print(parents)

    return parents


def generate_matrix_np(df: pd.DataFrame, location_x=0., location_y=0., step_x=3., step_y=3., shape=(20, 20), row=5):
    # Create Numpy Array from dict
    gen_nr = max(df['gen'])
    matrix = df.to_numpy()
    matrix = matrix[:, 1:]
    matrix = matrix.reshape((row, row, 6))

    dict = {}

    # Add index to keep track of item
    index = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            obj = sshape.make_bpy_mesh(shape, name=f'nr_{index}', weld=True)

            # Create dictionary for to mutate the next generation
            dict[f'nr_{index}'] = {f'gen': gen_nr, 'm': matrix[i, j, 0], 'a': matrix[i, j, 1], \
                                   'b': matrix[i, j, 2], 'n1': matrix[i, j, 3], 'n2': matrix[i, j, 4], \
                                   'n3': matrix[i, j, 5]}
            df = pd.DataFrame(dict).T

            x, y, z = sshape.supercoords(
                # m, a, b, n1, n2, n3 (1x6 or 2x6)
                matrix[i, j, :],
                # u,v resolution
                shape=(shape)
            )
            sshape.update_bpy_mesh(x, y, z, obj)

            # Shift location for matrix positioning
            obj.location.x = location_x
            obj.location.y = location_y
            location_x += step_x

            index += 1

            if j == matrix.shape[1] - 1:
                location_x = 0

        location_y += step_y

    df.to_pickle(bpy.path.abspath(bpy.path.abspath("//" + "dataframe.pkl")))


def remove_meshes():
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    print("All meshes are removed")


def run(from_file=False):
    if from_file:
        generate_ss_from_file(path=bpy.path.abspath("//" + "saved_selection.pkl"))
        set_height(2)

    else:
        # Store in variable
        objs = bpy.context.selected_objects

        if len(objs) > 0:

            parents = parent_selection(objs)
            survivors = crossover_mutate(parents, crossover=True, mutate=True)

            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            remove_meshes()  # remove meshes to reduce filesize

            generate_matrix_np(survivors)
            set_height(2)

        else:
            population = generate_population(size=(25))
            print("Population is: ", type(population))
            generate_matrix_np(population)
            set_height(2)


##### BFEORE RUNNING SCRIPT DELETE ALL OBJECTS FIRST ( PRESS a key and x or del ) #####
# Run the script with flag to create from file
# The first run will install and import the dependencies
if __name__ == '__main__': # run the script with flag to load selected objects from file.
    run(from_file=False)