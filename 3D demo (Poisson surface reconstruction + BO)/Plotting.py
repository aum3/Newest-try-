#Importing everything:
import numpy as np
import geometric_kernels
from geometric_kernels.spaces import Mesh
from geometric_kernels.kernels import MaternGeometricKernel, MaternKarhunenLoeveKernel
import matplotlib as mpl
import matplotlib.pyplot as plt
import kaleido
import plotly
import plotly.io as pio
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re 


#####################################

#1) Each method is to take in a mesh, trace and then output a trace. 
#2) Keep in mind it's certain I'm gonna forget this plotly shit so keep that in mind when writing this code. 
#3) The cleanse_unpack is to allow for forgetting the ** when inputting plotting kwargs. This easy fix must be
#removed and forgetting ** must raise an error in case it causes major issues downstream

#####################################





############CRASH COURSE ############
'''
1) Traces are akin to numpy's axes. Figs are the same
2) Doing multiple plots on the same fig is walkthroughed in the mesh.ipynb file. 
3) The below is enoough for plotting vectors of points onto a mesh and also adding hoverdata. Each function
takes in a trace (i.e axes) and outputs one so easy manipulation. 
4) Using plotly is quite easy except for the dict kwarg stuff which is unfamiliar. 
'''
#####################################

def fig_show(Fig, filename = None, update_fig = False, camera = None):
    '''
    This just wraps up plotting in a nice format bc I'm probably gonna forget how to do it. 


    update_figure is for using the update_figure given down below. 
    Camera is a 3-element dic of 3-element dics which specify the camera position. See below for how it works. 
    '''
    if True:
        fig = update_figure(Fig)
    if camera is not None:
        fig.update_layout(scene_camera = camera)
    else:
        camera = dict(
            up=dict(x=0, y=1, z=0), # which direction is  up
            center=dict(x=0, y=0, z=0), # point that the camera is looking at
            eye=dict(x=-2, y=1.3, z=2)) # positition of the camera. 
            #x is pointing towards eye , y is pointing up as usual.  z is the left-right axis.
    if filename is None:
        fig.update_layout(
            margin = {'t': 50},
        )
        fig.show()
        return 
    else:
        raise Exception("Saving not implemented yet")
def update_figure(fig, **kwargs): #no clue if kwargs would work here

    """Utility to clean up figure"""
    fig.update_layout(scene_aspectmode="cube")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # fig.update_traces(showscale=False, hoverinfo="none")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        )
    )
    fig.update_layout(**kwargs)
    return fig
def plot_mesh(mesh, vertices_colors = None, marker_kwargs = {}, **trace_def_kwargs):
    "returns the traces of a proper mesh with filled-in faces"
    '''
    MARKER-KWARGS ARE KWARGS THAT GO INSIDE THE MARKER DICT
    '''
    
    ret_trace = go.Mesh3d(
        x=mesh.vertices[:, 0], #the x-values of the vertices 
        y=mesh.vertices[:, 1], #the y-values of the vertices
        z=mesh.vertices[:, 2], #the z-values of the vertices
        i=mesh.faces[:, 0],  #the first index of each face
        j=mesh.faces[:, 1], # the second index of each face
        k=mesh.faces[:, 2], # the third index of each face
        colorscale='viridis',
        intensity=vertices_colors,
        **trace_def_kwargs
    )
    return ret_trace
def add_custom_hover_data(trace, **kwargs):
    """
    Parameters:
    - trace: The Plotly trace to which the custom data will be added.
    - plotting_kwargs. The TOTAL kwargs as if you were defining a trace. This function handles all the horseshit. Nothing except the 
    hovertemplate or the customdata is looked at 


    Returns:
    - The updated trace with custom hover data. WORKS EVEN WHEN THE ORIGINAL TRACE HAS HOVER DATA
    """
    plotting_kwargs = kwargs.copy()
    plotting_kwargs = cleanse_unpack(plotting_kwargs)
    
    
    import copy 
    custom_data = plotting_kwargs.get('customdata', None).copy()
    if custom_data is None: return trace
    if custom_data.ndim != 2: 
        raise Exception("definitely needs to be 2D. Perhaps change its dimension permanently below this line ?")
    

    num_points_in_the_trace = len(trace.x)
    # if (custom_data.shape[0] != num_points_in_the_trace) is True: #TODO: The index was 1 before. changed it 
    #     custom_data = custom_data.T
    #     if custom_data.shape[1] != num_points_in_the_trace:
    #         raise Exception("custom_data is not of correct dimension mate")
        

    custom_data.reshape(num_points_in_the_trace, -1)
    if custom_data is None: #handling error case
        return trace  
    

    #Merging the customdata 
    old_custom_data = trace.customdata
    if old_custom_data is None:
        trace.customdata = custom_data
    else:
        if trace.customdata.shape[0] != custom_data.shape[0]:  #error case
            raise Exception("what the hell are you doing sir? custom_data supplied is not of correct dimension")
        
        
        trace.customdata = np.hstack([trace.customdata, custom_data])



        
        
    
    #UPDATING (MERGING) HOVERDATA
    old_hover = "" if trace.hovertemplate is None else trace.hovertemplate 
    addition_hover = plotting_kwargs.get('hovertemplate', "")
    

    #Creating nicely-formatted hovertemplate in the case one isn't provided 
    if addition_hover == "":
        num_columns = custom_data.shape[1]
        template_parts = [f"Value {i+1}: %{{customdata[{i}]:.4f}}<br>" for i in range(num_columns)]
        new_bit = "".join(template_parts)
        new_bit += "<extra></extra>"
        addition_hover += new_bit
        addition_hover += "<br>" #causing issues i reckon!!!!!!!! TODO

    ###Crazy check to ensure that old_hover contains a <br> at the end (or else the hover window becomes weird) 
    
    if addition_hover and not re.search(r"<br>\s*$", addition_hover):
        addition_hover += "<br>"
    if old_hover and not re.search(r"<br>\s*$", old_hover):
        old_hover += "<br>"

    trace.hovertemplate = old_hover + addition_hover
        
    return trace

def cleanse_unpack(erroneous_dict):
    #If kwargs are passed in as a dict without unpacking then this function needs to be used 
    if len(erroneous_dict) == 1 and isinstance(list(erroneous_dict.values())[0], dict) and list(erroneous_dict.keys())[0] != "marker":
        erroneous_dict = list(erroneous_dict.values())[0]

    # Your function logic using the 'kwargs' dictionary
    return erroneous_dict

def vector_values_to_mesh_trace(mesh, values, **plotting_kwargs):
    '''
    Remember that plotting_kwargs is to contain every plotting kwarg aswell as a marker kwarg dictionary within it. This is optional

    First we remove the marker-level kwargs to get one trace-level and one marker-level. Thus no monkey shit. 
    '''

    values = values.ravel() 
    kwargs = plotting_kwargs.copy()
    kwargs = cleanse_unpack(kwargs)
    


    marker_update = {'color': values, 'colorscale': 'Viridis'}  #get rid of colorscale sonon
    if kwargs.get('marker', None) is not None:  #If a marker is supplied
        kwargs['marker'].update(marker_update)
    else:
        kwargs['marker'] = marker_update

        
        
    
    
    
    if True: #Boilerplate stuff for dictionary handling. 
        values = np.ravel(values) #Apparently needs to be 0d vector.maybe this is wrong and it should be a row vector?
        custom_data = kwargs['marker'].get('customdata', None)
        if custom_data is not None:
            if custom_data.shape[0] != mesh.num_vertices:
                raise Exception("hoverdata should be put into the proprietary function to avoid putting in custom data with no rendering of it")
    
    trace = go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        mode='markers+lines', #TODO: changed this since the last time and haven't checked for errors.
        **kwargs)
    return trace




#EXAMPLE FUNCTION:
def plot_couple_vectors_onto_fig(mesh, vector1, fig = None, vector2 = None, vector3 = None, colorscales = None, trace_names = None,  **plot_kwargs):
    '''

    This function is not really useful. Just for quickly bashing vector mesh data onto the data
    without being able to actually alter things about the traces. But it's illustrative of how to use this plotly crap. 




    colorscales is a list of strings of colorscale names you want in order.

    plotting_kwargs is the kwargs which ends up getting inside a definition of each of the traces. This function gives to each 
    trace the same kind of data. Needs a moderate level change in order to give each trace a different plot_kwargs. 

    '''
    if trace_names == None:
        trace_names = ['1', '2', '3']
    
    if True: #boilerplate stuff (remove Nones from vector; ensure fig exists):
        
        vectorlist = [x.ravel() for x in [vector1, vector2, vector3] if x is not None]
        if fig is None:
            fig = go.Figure()
        for vector in vectorlist: 
            if vector is None:
                vectorlist.remove(vector)
                continue
            else:
                vector = vector.reshape(-1,1)
                assert vector.shape[0] == mesh.vertices.shape[0], "vector should have the same number of rows as there are vertices in the mesh"

            

    
    
    if colorscales is None:   #Colourscheme setting. Many options so see inside
        Decent_duochromatics = ['Reds', 'OrRd', 'BuGn', 'Tealgrn', 'thermal', 'hot']
        Decent_trichromatics = ['Bluered', 'thermal', 'inferno', ]
        colorscales = ['Tealgrn', 'Tealgrn', 'Tealgrn' ] #Each trace must have its own colorscale or else plotly get's confused.

    traces_list = []
    for i, vector in enumerate(vectorlist):
        
        
        '''
        CAREFUL: All the mellarky below is because the plotting arguments is given as a kwargs dict wherein there's
        an inner dict called the marker. The marker itself has yet another inner dict called colorbar. The 
        dictionary updating is carefully done so that this plotting function overrides any plotting parameters that
        are put in when the function is called. 
        '''

        
        kwargs = plot_kwargs.copy() 
        kwargs = cleanse_unpack(kwargs)
        # if plot_kwargs.get('plot_kwargs', False):
        #     kwargs = kwargs['plot_kwargs']



        #UPDATE TRACE-LEVEL INFO
        kwargs_update = {'name' : trace_names[i]}
        kwargs.update(kwargs_update)  #delete these 2 if it doesn't work 


        #UPDATE MARKER (SUBDICT OF PLOTTING KWARGS)
        kwargs.setdefault('marker', {}) #fancy function sets a key's value if it doesn't already exist
        marker_dict = kwargs['marker'] 
        marker_update = {'colorscale': colorscales[i]}
        marker_dict.update(marker_update)

        #UPDATE COLORBAR (SUBDICT OF MARKER)
        left_shunt = i*0.1        
        marker_dict.setdefault('colorbar', {})
        colorbar_update = {'x' : 1 - left_shunt, 'title' : trace_names[i]}
        marker_dict['colorbar'].update(colorbar_update)

        

        #CREATING NEW TRACE
        new_trace = vector_values_to_mesh_trace(mesh, vector, plotting_kwargs = kwargs)
        # new_trace = add_custom_hover_data(new_trace, plotting_kwargs = kwargs)
        traces_list.append(new_trace)
        fig.add_trace(new_trace)
    
    return fig


    
    
    
    
    
        



