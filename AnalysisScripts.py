import numpy as np, sys, time, inspect, os, pickle

# def gaussian_to_pdb(gaussian_log, pdb_file):

# THIS HAS NOT YET BEEN DEVELOPED.
#     with open(gaussian_log, 'r') as log, open(pdb_file, 'w') as pdb:
#         write_flag = False
#         atom_number = 1
#         for line in log:
#             if 'Standard orientation:' in line:
#                 write_flag = True
#                 log.readline()  # Skip the next 4 header lines
#                 log.readline()
#                 log.readline()
#                 log.readline()
#             elif write_flag and '---------------------------------------------------------------------' in line:
#                 break  # End of the atom list
#             elif write_flag:
#                 fields = line.split()
#                 atom_type = int(fields[1])  # Atomic number
#                 x = float(fields[3])
#                 y = float(fields[4])
#                 z = float(fields[5])
#                 element = atomic_number_to_element(atom_type)
#                 pdb.write(f"ATOM  {atom_number:5d}  {element:2s}   MOL     1     {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
#                 atom_number += 1

def collect_atoms_in_cluster_distribution_from_directory(directory):
    atoms_in_cluster_distribution = []

    # List all .pkl files in the directory
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        
        # Load the pickle file
        with open(file_path, 'rb') as file:
            frame_dict = pickle.load(file)

        # Extract the atoms_in_cluster values
        for frame_number, clusters in frame_dict.items():
            for cluster_number, (atoms_in_cluster, _) in clusters.items():
                atoms_in_cluster_distribution.append(atoms_in_cluster)

    return atoms_in_cluster_distribution

def atomic_number_to_element(atomic_number):
    periodic_table = {1: 'H', 6: 'C', 8: 'O', 7: 'N', 16: 'S'}  # Extend this dictionary as needed
    return periodic_table.get(atomic_number, 'X')

def progressbar(it, prefix="", size=60, out=sys.stdout):
    #Displays a progress bar as output for functions which are performed within a for loop.
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)        
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0 
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def make_three_tuples(n1s, n2s, n3s):
    # Inputs: 
    #   - ns: integers
    # Outputs:
    #   - List of tuples (a, b, c) for all a in [n1min, n1max), [n2min, n2max), [n3min, n3max)
    # Example: make_three_tuple([1,2,3], [3,4], [6,8]) = [[1,3,6], [1,3,8], [1,4,6],...]

    triples = []
    for i in n1s:
        for j in n2s:
            for k in n3s:
                triples.append((i,j,k))

    return triples

def distance(a, b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def extract_clusters_from_frame(frames, dump_path, rc, frame_numbers):
    lx, ly, lz = get_box_lengths_from_dump_file(dump_path=dump_path)

    clusters_by_frames = {}; cluster_index = 1; clusters = {}; current_cluster = []; reference_atoms=[]; queue1 = []; queue2 = []; clusters_by_frames = {}

    for N in frame_numbers:
        unclustered_atoms=frames[N].copy(); 
        while unclustered_atoms!={i:[] for i in frames[N].keys()}:
            remaining_molecules = []
            for i in unclustered_atoms.keys():
                for j in range(len(unclustered_atoms[i])):
                    remaining_molecules.append(unclustered_atoms[i][j][2])
            remaining_molecules=set(remaining_molecules)

            reference_molecule=min(remaining_molecules)
            for i in list(unclustered_atoms.keys()):
                for j in range(len(unclustered_atoms[i])):
                    if unclustered_atoms[i][j][2]==reference_molecule:
                        reference_atoms.append((i,unclustered_atoms[i][j])) 
                        queue1.append((i, unclustered_atoms[i][j]))
                        current_cluster.append((i, unclustered_atoms[i][j]))

            for i in queue1:
                unclustered_atoms[i[0]].remove(i[1])

            while queue1!=[]:
                for i in range(len(queue1)): 
                    # For every atom that is within rc of some atom on the reference molecule, search for other atoms that are within rc of it (but not on the reference molecule)
                    x, y, z = queue1[0][1][3:] 
                    search_subregions = create_subregion_queue_from_location(x, y, z, lx, ly, lz, rc)
                    for subregion in search_subregions:
                        for atom in unclustered_atoms[subregion]:
                            if distance(atom[3:], (x, y, z)) < rc:
                                for j in list(unclustered_atoms.keys()):
                                    for k in unclustered_atoms[j]:
                                        if k[2]==atom[2]:
                                            queue2.append((j,k))
                                            unclustered_atoms[j].remove(k)
                    queue1.remove(queue1[0])

                for atom in queue2:
                    if atom not in queue1:
                        queue1.append(atom)
                        current_cluster.append(atom)
                
                queue2=[]

            clusters[cluster_index]=(len(current_cluster), current_cluster)
            current_cluster=[]; cluster_index+=1 

        sorted_clusters = dict(sorted(clusters.items(), key=lambda item: item[1][0], reverse=True))
        clusters_by_frames[N]=sorted_clusters

    with open('../pickle_files/frames_'+str(min(frame_numbers))+'_'+str(max(frame_numbers))+'_clusters.pkl','rb') as doc:
        pickle.dump(clusters_by_frames, doc)

    return clusters_by_frames

def map_position_into_box_index(x, y, z, lx, ly, lz, Nboxes):
    dx = lx/Nboxes; dy = ly/Nboxes; dz = lz/Nboxes
    x = int(x//dx); y = int(y//dy); z = int(z//dz)
    return (x, y, z)

# def extract_clusters_from_frame(args):
    dump_path, data_path, rc, atom_types, N = args

    file_name = 'frame_'+str(N)+'_clusters.pkl'

    frame = read_frames_from_dump_file(dump_path=dump_path, data_path=data_path, atom_types=atom_types, N=N)
    lx, ly, lz = get_box_lengths_from_dump_file(dump_path=dump_path)

    clusters_by_frames = {}; cluster_index = 1; clusters = {}; current_cluster = set(); reference_atoms=set(); queue1 = set(); queue2 = set()
    unclustered_atoms=frame[N].copy(); shell_index=0
    # Extract a reference molecule from the unclustered atoms list. We add this molecule's atoms to a queue and we will calculate all atoms which are within rc of these 

    while unclustered_atoms!=dict({}):
        reference_molecule = min([unclustered_atoms[i][1] for i in unclustered_atoms.keys()])
        print('Reference Molecule:', reference_molecule)
        for i in list(unclustered_atoms.keys()):
                if unclustered_atoms[i][1]==reference_molecule:
                    reference_atoms.add((i, unclustered_atoms[i])); queue1.add((i, unclustered_atoms[i])); current_cluster.add((i, unclustered_atoms[i]))

        for i in queue1:
            del unclustered_atoms[i[0]]

        while queue1!=set():
            queue1=list(queue1)
            for i in range(len(queue1)):
                for j in unclustered_atoms.keys():
                    if distance_with_pbc(queue1[0][1][2:], unclustered_atoms[j][2:], lx, ly, lz) < rc:
                        for k in unclustered_atoms.keys():
                            if unclustered_atoms[k][1]==unclustered_atoms[j][1]:
                                queue2.add((k, unclustered_atoms[k]))
                current_cluster.add(queue1[0]); 
                try: 
                    del unclustered_atoms[queue1[0][0]];
                except:
                    pass
                queue1.pop(0); 
            queue1=set(queue1); reference_atoms=set()

            for i in queue2:
                queue1.add(i)

            queue2=set()
            print('Next Layer')
            print(len(unclustered_atoms),'atoms remaining')
        
        clusters[cluster_index]=(len(current_cluster),current_cluster)
        current_cluster=set(); cluster_index+=1; shell_index=0
        print('Beginning New Cluster')

    clusters = dict(sorted(list(clusters.items()), key = lambda item: item[1][0], reverse = True)); clusters2={} #Sort the clusters by size and re-index so that the largest cluster is cluster 1

    for i, j in enumerate(clusters):
        clusters2[i]=clusters[j]
        
    clusters_by_frames[N] = clusters2

    with open(file_name,'w') as doc:
        pickle.dump(clusters_by_frames, doc)
    doc.close()

    return clusters_by_frames

def convert_all_atom_dump_to_COM_dump(data_path, dump_path, COM_path, mol_count, calculate_vCOM=False):
        #data_path = string; dump_path = string; COM_path = string; mol_count = list of (molecule_name (string), molecule_count) tuples appearing in the order they appear in the data file
        #mol_count example: [('TFSI', 100), ('Li', 100), ('Water', 5000)] <<< TFSI atoms are listed first, then Li, then water 

        #Reduces an all-atom dump file to a COM dump file. Takes in a lammps data file, a dump file, and a directory at which to write the center-of-mass dump file. Make sure the atoms/molecules in your data file are sorted by id number. This will not be the case if you are loading a data file which you've created from a restart file. 

    print('Converting all atom dump file at '+dump_path+' to center-of-mass dump file at '+COM_path)
    import numpy as np

    #Extract the location of all the masses of the atoms in the simulation by identifying the "Masses" section of the LAMMPS data file
    with open(data_path,'r') as data_file: 
        print('Reading data file...')
        masses={}
        lines=data_file.readlines()
        for i in range(len(lines)):
            if lines[i]=='Masses\n':
                masses_start=i
                break

    #Identify the "Atoms" section within the LAMMPS data file. This section contains the types of each atom in the simulation. With this information, we can connect each atom type with an associated mass for calculation of the COM  
        for i in range(len(lines)):
            if lines[i]=='Atoms\n':
                atoms_start=i 
                break
        for i in range(len(lines)):
            if lines[i]=="Bonds\n":
                bond_start=i
                break

        #After having identified where the "Atoms" and "Masses" sections of the LAMMPS data file are, write the information to dictionaries. Now we can look up the mass, and molecule ID of each atom within the $masses and $mols dictionaries.
            
        mass_lines=lines[masses_start+2:atoms_start-1]; atom_lines=lines[atoms_start+2:bond_start-1]
        
        for i in range(len(mass_lines)):
            masses[int(mass_lines[i].strip().split()[0])]=float(mass_lines[i].strip().split()[1])

        #Initialize a 'mols' molecule dictionary which includes the molecule number, atom numbers of all atoms in the molecule, and the name of the molecule, which is taken from mol_count
                
        mols={i: [] for i in range(1, int(lines[bond_start-2].strip().split()[1])+1)} 

        for i in range(len(atom_lines)):
            mols[int(atom_lines[i].strip().split()[1])].append(int(atom_lines[i].strip().split()[0]))
        nmols=len(mols.keys()); print(f'{nmols}'+' molecules identified in data file')

    names = []
    for i in range(len(mol_count)):
        for j in range(mol_count[i][1]):
            names.append(mol_count[i][0])

    for i in range(len(names)):
        try:
            mols[i+1].append(names[i]) #Python indexes from 0, but molecule number is indexed from 1.
        except:
            raise ValueError('Given molecule counts may not match data file')

    #Read in dump file, use mols dictionary and masses dictionary to calculate COM of each molecule, write results to the specified directory
    with open(dump_path,'r') as dump_file:
        with open(COM_path, 'w') as COM_file:
            print('Reading dump file...')
            lines=dump_file.readlines(); natoms=int(lines[3]); attributes=lines[8].strip().split()
            if 'xu' not in attributes or 'yu' not in attributes or 'zu' not in attributes:
                raise ValueError('Dump file must contain unwrapped trajectories for reliable COM calculation') 
            
            attributes_string='Dump file contains '
            for i in attributes[2:]: #First two entries in the ITEM: ATOMS lines are "ITEM:" and "ATOMS" => not necessary to load
                attributes_string+=i+' '
            print(attributes_string)

            print(f'{natoms} atoms identified in dump file...')
            
            print('Writing COM file...')
            i=0
            while i<len(lines):
                if lines[i]=='ITEM: TIMESTEP\n':
                    for j in range(3):
                        COM_file.write(lines[i]); i+=1
                    COM_file.write(str(len(mols))+'\n'); i+=1
                    for j in range(5):
                        COM_file.write(lines[i]); i+=1
                    atoms=sorted(lines[i: i+natoms], key=lambda line: int(line.split()[0]))
                    for k in list(mols.keys()):
                        rCOM=np.array([0,0,0],dtype=float);  M=0; vCOM=np.array([0,0,0],dtype=float);
                        for l in mols[k][:-1]: 
                            atom_line=atoms[l-1].strip().split(); atom_type=int(atom_line[1]); m=masses[atom_type] #Again, since atom_line loads in atoms indexed from 0 to natoms-1 but lammps records atom numbers from 1 to natoms, we shift the index
                            x=float(atom_line[attributes.index('xu')-2]); y=float(atom_line[attributes.index('yu')-2]); z=float(atom_line[attributes.index('zu')-2]); rCOM+=m*np.array([x,y,z]); 
                            if calculate_vCOM==True:
                                vx=float(atom_line[attributes.index('vx')-2]); vy=float(atom_line[attributes.index('vy')-2]); vz=float(atom_line[attributes.index('vz')-2]);  vCOM+=m*np.array([vx,vy,vz]); 
                            M+=m
                        rCOM=rCOM/M; vCOM=vCOM/M
                        if calculate_vCOM==False:
                            COM_file.write(f'{k} '+f'{mols[k][-1]} '+f'{rCOM[0]} '+f'{rCOM[1]} '+f'{rCOM[2]}\n')
                        elif calculate_vCOM==True:
                            COM_file.write(f'{k} '+f'{mols[k][-1]} '+f'{rCOM[0]} '+f'{rCOM[1]} '+f'{rCOM[2]} '+f'{vCOM[0]} '+f'{vCOM[1]} '+f'{vCOM[2]}\n')
                    i+=natoms
        COM_file.close()
    dump_file.close()
    print('Complete!')

def remove_frame_from_dump_file(input_dump, output_path, N):
    #input_dump = string; output_path = string; N = int (frame to be removed)

    #Takes in a dump file and removes the (N)th frame of data from the file. This is useful because when restarting a simulation from a restart file in LAMMPS, the last frame of the first simulation is printed to the second dump file. If you then concatenate the two dump files together, the last frame of the first simulation will be repeated. This can affect the calculation autocorrelation functions, time-dependent quantities, etc, and therefore the redundant frame must be removed before calculating properties.

    #NOTE: This script presumes that the first frame in your simulation is the "zeroth" timestep, ie the initial condition of the file

    with open(input_dump, 'r') as file, open(output_path, 'w') as path:
        print('Reading input file at '+f'{input_dump}')
        lines = file.readlines(); natoms = int(lines[3])
        exclude=range(N*(natoms+9),(N+1)*(natoms+9))
        print('Input file loaded, frame data extracted. Writing data to new file at ' +f'{output_path}')
        for i in range(len(lines)):
            if i not in exclude:
                path.write(lines[i])
    file.close(); path.close()
    print('Complete!')

def sort_dump_file_frames_by_atom_ID(input_dump, output_path):
    print('Sorting atoms in dump file frames at '+ input_dump)
    #Takes in a dump file and sorts each frame by atom ID. This is useful when used in conjunction with LiquidLib, which can also take in a "mol file" which tells LiquidLib which atoms are in which molecules. However, LiquidLib demands that the atoms in the "mol file" appear in the same order as in the dump file. Hence, it's useful to order dump files by atom ID.

    print('Reading dump file...')
    with open(input_dump, 'r') as file, open(output_path, 'w') as path:
        lines = file.readlines(); natoms = int(lines[3])
        print('Dump file loaded. Sorting frames by atom ID')
        i = 0
        while i < len(lines):
            if lines[i].startswith('ITEM: TIMESTEP'):
                atoms_data=[]
                #Identify the header section of each timestep and write these lines to the new file
                for j in range(9):
                    path.write(lines[i]); i+=1
                
                #Now that we have written the header portion of the file, load in all the atom information and sort it.
                for j in range(natoms):
                    atoms_data.append(lines[i+j])
                atoms_data = sorted(atoms_data, key = lambda element: int(element.split()[0]))

                #Write the sorted atom data to the output file path and then skip forward natom lines, to the next TIMESTEP line.
                for j in range(len(atoms_data)):
                    path.write(atoms_data[j]); i+=1
    file.close(); path.close()
    print('Complete!')

def search_for_repeated_timestep_in_dump_file(file_path):
    #Takes in a dump file at file_path and will find any repeated timesteps in the simulation
    print("Searching for repeated timesteps in "+file_path)
    with open(file_path,'r') as doc:
        print('Reading dump file ...')
        lines=doc.readlines()
        timesteps=[]
        for i in range(len(lines)):
            if lines[i].startswith('ITEM: TIMESTEP'):
                if lines[i+1] in timesteps:
                    print('Found repeated timestep: '+lines[i+1])
                    timesteps.append(lines[i+1])
                else:
                    timesteps.append(lines[i+1])
    print('Complete!')

def remove_data_from_dump_file(input_filename, output_filename, remove):
# remove is a list of strings corresponding to entries in the ITEM: ATOMS lines of the dump file. These are the data attributes that you'd like to remove from your dump file. EX: remove = ['vx', 'vy', 'vz']
    
    intro='Removing '
    for i in remove:
         intro+=i+' '; 
    intro+='data from '+input_filename; print(intro); print('Reading dump file')

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        lines=infile.readlines(); data_attributes = lines[8].strip().split()[2:]; indices=[]; new_attributes = lines[8]; natoms = int(lines[3])
        print('Dump file loaded, now removing data')
        for i in remove:
            if i in data_attributes:
                indices.append(data_attributes.index(i))
            elif i not in data_attributes:
                raise ValueError(i + ' is not found in dump file data directory. Check ITEM: ATOMS lines to verify what information is contained in dump file.')
        for i in remove:
            new_attributes=new_attributes.replace(i,'').strip()

        i=0
        while i < len(lines):
            if lines[i].startswith('ITEM: TIMESTEP'):
                for j in range(8):
                    outfile.write(lines[i]); i+=1
                outfile.write(' '.join(new_attributes.split())+'\n'); i+=1
                for j in range(natoms):
                    holder = lines[i].strip().split()
                    for index in sorted(indices, reverse=True):
                        del holder[index]
                    outfile.write(' '.join(holder)+'\n'); i+=1
    infile.close(); outfile.close(); print('Complete!')

def get_box_lengths_from_dump_file(dump_path):
# Designed to read in only minimal portion of dump file.
    lines=[]
    with open(dump_path) as dump_file:
        for _ in range(5):
            next(dump_file)
        for _ in range(3):
            line = dump_file.readline(); lines.append(line.strip().split())
        x1, x2 = np.array(lines[0], dtype=float); lx = x2-x1
        y1, y2 = np.array(lines[1], dtype=float); ly = y2-y1
        z1, z2 = np.array(lines[2], dtype=float); lz = z2-z1
    return (lx, ly, lz)

def write_LL_molecule_file(data_file, molecule_dict, output_path):
    #data_file is a path to a data file which will be read to extract the molecule ID for a given atom. molecule_dict is a dictionary for which the keys are the names of the molecules in your simulation and the values are the number of ATOMS in molecules with this name. For example, molecule_dict = {'methanol':12, 'water': 18} indicates that there are 2 methanol molecules (each with 6 atoms) to start your data file, and then 6 water molecules (each with 3 atoms) following that. output_path is the location of the molecule file you'd like to create

    molecule_names = []
    for i in molecule_dict.keys():
        for j in range(molecule_dict[i]):
            molecule_names.append(i)

    with open(data_file,'r') as data:
        molecule_ids=[]
        data_lines=data.readlines()
        natoms = int(data_lines[2].strip().split()[0])
        for i in range(len(data_lines)):
            if data_lines[i]=='Atoms'+'\n':
                atoms_start = i+2
                break
        for i in range(len(data_lines)):
            if data_lines[i]=='Bonds'+'\n':
                atoms_end=i-1
                break
        for i in range(atoms_start, atoms_end):
            molecule_ids.append(data_lines[i].strip().split()[1])
        data.close()

    with open(output_path,'w') as output:
        if len(molecule_ids)!=len(molecule_names):
            raise ValueError('Number of atoms does not match number of molecule names provided.')
        
        output.write('Molecule file generated for LiquidLib input from data file at '+str(data_file)+'\n')
        output.write(str(natoms)+'\n')
        for i in range(len(molecule_ids)):
            output.write(molecule_ids[i]+' '+molecule_names[i]+'\n')
        output.close()
    print('Molecule file written to '+output_path)


def distance_with_pbc(a, b, lx, ly, lz):
    # Inputs:
    #   - a and b are 3-tuples/lists that contain the positions to be wrapped are input coordinates to be wrapped
    #   - lx, ly, lz are box lengths

    # Outputs:
    #   - distance (float) between points a and b with respect to the minimum image convention for the given pbcs.

    dx = b[0] - a[0]; 
    if dx < -lx/2:
        dx+=lx
    elif dx >= lx/2:
        dx-=lx

    dy = b[1] - a[1]
    if dy < -ly/2:
        dy+=ly
    elif dy >= ly/2:
        dy-=ly

    dz = b[2] - a[2]
    if dz < -lz/2:
        dz+=lz
    elif dz >= lz/2:
        dz-=lz
    return np.sqrt(dx**2 + dy**2 + dz**2)

def wrap_coordinates_into_box(x, y, z, lx, ly, lz, x0=0, y0=0, z0=0):
    # Inputs:
    #   - x, y, z are input coordinates to be wrapped
    #   - lx, ly, lz are box lengths
    #   - x0, y0, z0 are usually the origin, but if your box is not at the origin, input the corner closest to the origin

    # Outputs:
    #   - x, y, and z locations after the point has been wrapped onto the box

    x=((x-x0)%lx)+x0; y=(y-y0)%ly+y0; z=(z-z0)%lz+z0
    return (x, y, z)

def read_dump_file_for_clustering(dump_path, data_path, atom_types, frame_numbers, Nboxes=10, write_results = False, write_path = ''):
#  Inputs: 
#   - dump_path: path (str) to dump file which will be read
#   - data_path: path (str) to LAMMPS data file
#   - atom_types: list of atom types (strs) that you want to include in the dataset. For example, you can omit reading in oxygens from a water simulation by using atom_types = ['H'], which will only read in atoms of type 'H' from your LAMMPS file. However, LAMMPS atom types are typically '1', '2', etc.
#   - frame_numbers: list of frame numbers (ints) to be read in
#   - Nboxes: divide the simulation into Nboxes by Nboxes by Nboxes regions

#  Outputs:
#   - frames (dict) of form frames[frame_number][(region_index_3_tuple)] = [atom_type, molecule_number, x, y, z]

# Other notes:
#   - Use this function to create a pickle file that will be used for the "extract_clusters..." function.

    with open(dump_path) as dump_file, open(data_path) as data_file:
        lx, ly, lz = get_box_lengths_from_dump_file(dump_path)

        data_lines=data_file.readlines(); 
        natoms=int(data_lines[2].strip().split()[0]); 

        for i in range(len(data_lines)):
            if data_lines[i]=='Atoms\n':
                atoms_start=i 
                break
        for i in range(len(data_lines)):
            if data_lines[i]=="Bonds\n":
                bonds_start=i
                break
        atom_lines = []; mols = {}

        for line in data_lines[atoms_start+2: bonds_start-1]:
            if line.strip().split()[2] in atom_types:
                atom_lines.append(line)

        for i in range(len(atom_lines)):
            mols[int(atom_lines[i].strip().split()[0])]=int(atom_lines[i].strip().split()[1])
        data_file.close()

        frames = {}

        for _ in range(8):
            next(dump_file)
        attributes = dump_file.readline().strip().split(); dump_file.seek(0)

        advance = (natoms+9)*np.array(frame_numbers)
        for i in range(1, len(advance)):
            advance[i]=advance[i]-((frame_numbers[i-1]+1)*(natoms+9))

        for i1 in range(len(advance)):
            frames[frame_numbers[i1]]={}
            for i2 in range(advance[i1]):
                next(dump_file)
            for i3 in range(9):
                next(dump_file)
            for i4 in make_three_tuples(range(Nboxes), range(Nboxes), range(Nboxes)):
                frames[frame_numbers[i1]][i4]=[]

            for j in range(natoms): #Note that .readline() function automatically skips to next line, so we do not need to index over j. Just call it natoms times, and .readline() will advance forward in the document automatically
                info=dump_file.readline().strip().split()
                if info[1] in atom_types:
                    name = int(info[attributes.index('id')-2]); species = str(info[attributes.index('type')-2])
                    mol_number = mols[name]
                    x = float(info[attributes.index('xu')-2]); y = float(info[attributes.index('yu')-2]); z = float(info[attributes.index('zu')-2])
                    x, y, z = wrap_coordinates_into_box(x, y, z, lx, ly, lz); 
                    index_x, index_y, index_z = map_position_into_box_index(x, y, z, lx, ly, lz, Nboxes=Nboxes)
                    try:
                        frames[frame_numbers[i1]][(index_x,index_y,index_z)].append([name, species, mol_number, x, y, z])
                    except:
                        print('Error!:, failed to append line'+str(f'{info}') +' to file') #Frame dictionary now contains region_index: [atom_number, species, molecule number, wrapped location] data for all of the atoms of the selected type(s)
        dump_file.close()

    if write_results == True:
        with open(write_path, 'wb') as doc:
            pickle.dump(frames, doc)

    return frames

def read_frames_from_dump_file(dump_path, data_path, atom_types, frame_numbers, wrap = False):
    # Inputs: 
    #   - dump_path = path (str) to dump file which will be read
    #   - data_path = path (str) to LAMMPS data file
    #   - atom_types = list of atom types (strs) that you want to include in the dataset. For example, you can omit reading in oxygens from a water simulation by using atom_types = ['H'], which will only read in atoms of type 'H' from your LAMMPS file. However, LAMMPS atom types are typically '1', '2', etc.
    #   - frame_numbers = list of frame numbers (ints) to be read in
    # wrap = boolean of whether to wrap atomic positions back into pbc of the box or leave them as is. For clustering algorithms, wrap the trajectories. For g(r) or MSD calculations, don't wrap. 
    
    # Outputs:
    #   - frames (dict) of form frames[frame_number][atom_number] = [atom_type, molecule_number, x, y, z]

    
    with open(dump_path) as dump_file, open(data_path) as data_file:
        lx, ly, lz = get_box_lengths_from_dump_file(dump_path)

        data_lines=data_file.readlines(); 
        natoms=int(data_lines[2].strip().split()[0]); 

        for i in range(len(data_lines)):
            if data_lines[i]=='Atoms\n':
                atoms_start=i 
                break
        for i in range(len(data_lines)):
            if data_lines[i]=="Bonds\n":
                bonds_start=i
                break
        atom_lines = []; mols = {}

        for line in data_lines[atoms_start+2: bonds_start-1]:
            if line.strip().split()[2] in atom_types:
                atom_lines.append(line)

        for i in range(len(atom_lines)):
            mols[int(atom_lines[i].strip().split()[0])]=int(atom_lines[i].strip().split()[1])
        data_file.close()

        frames = {}
        for N in frame_numbers:
            dump_file.seek(0)
            frames[N]={}
            for _ in range(8):
                next(dump_file)
            attributes = dump_file.readline().strip().split(); dump_file.seek(0)

            for i in range(N*(natoms+9)+9):
                next(dump_file)

            for j in range(natoms):
                info=dump_file.readline().strip().split()
                if info[1] in atom_types:
                    name = int(info[attributes.index('id')-2]); species = str(info[attributes.index('type')-2])
                    mol_number = mols[name]
                    x = float(info[attributes.index('xu')-2]); y = float(info[attributes.index('yu')-2]); z = float(info[attributes.index('zu')-2])
                    if wrap==True:
                        x, y, z = wrap_coordinates_into_box(x, y, z, lx, ly, lz)
                        frames[N][name] = [species, mol_number, x, y, z] #Frame dictionary now contains atom_number: [species, molecule number, wrapped location] data for all of the atoms of the selected type(s)
                    if wrap==False:
                        frames[N][name] = [species, mol_number, x, y, z] #Frame dictionary now contains atom_number: [species, molecule number, location] data for all of the atoms of the selected type(s)

        dump_file.close()
        return frames

def collect_atoms_in_cluster_distribution_from_directory(directory):
    atoms_in_cluster_distribution = []

    # List all .pkl files in the directory
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        
        # Load the pickle file
        with open(file_path, 'rb') as file:
            frame_dict = pickle.load(file)

        # Extract the atoms_in_cluster values
        for frame_number, clusters in frame_dict.items():
            for cluster_number, (atoms_in_cluster, _) in clusters.items():
                atoms_in_cluster_distribution.append(atoms_in_cluster)
    return atoms_in_cluster_distribution

def create_subregion_queue_from_location(x, y, z, lx, ly, lz, rc, Nboxes=10):
    dx = lx/Nboxes; dy = ly/Nboxes; dz = lz/Nboxes
    
    x_index = (x%lx)//dx; y_index = (y%ly)//dy; z_index = (z%lz)//dz

    xmax = x+rc; xmin = x-rc; 
    ymax = y+rc; ymin = y-rc; 
    zmax = z+rc; zmin = z-rc; 
    
    xmax_index = int((xmax%lx)//dx); xmin_index = int((xmin%lx)//dx); 
    ymax_index = int((ymax%ly)//dy); ymin_index = int((ymin%ly)//dy); 
    zmax_index = int((zmax%lz)//dz); zmin_index = int((zmin%lz)//dz); 

    x_indices = []
    while x_index != (xmax_index+1)%Nboxes:
            x_indices.append(int(x_index)); x_index=(x_index+1)%Nboxes

    y_indices = []
    while y_index != (ymax_index+1)%Nboxes:
        y_indices.append(int(y_index)); y_index=(y_index+1)%Nboxes

    z_indices = []
    while z_index != (zmax_index+1)%Nboxes:
            z_indices.append(int(z_index)); z_index=(z_index+1)%Nboxes

    x_index = (x%lx)//dx; y_index = (y%ly)//dy; z_index = (z%lz)//dz

    while int(x_index) != (xmin_index-1)%Nboxes:
        x_indices.append(int(x_index)); x_index=(x_index-1)%Nboxes

    while y_index != (ymin_index-1)%Nboxes:
        y_indices.append(int(y_index)); y_index = (y_index-1)%Nboxes 

    while int(z_index) != (zmin_index-1)%Nboxes:
        z_indices.append(int(z_index)); z_index = (z_index-1)%Nboxes

    return make_three_tuples(set(x_indices), set(y_indices), set(z_indices))

if __name__=='__main__':
    functions = inspect.getmembers(__import__(__name__), inspect.isfunction)
    print("Functions:"+'\n')
    for name, _ in functions:
        if not name.startswith("__"): 
            print(name)

def read_lammps_log_file(log_path):
    data_dict = {}

    with open(log_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()] #Discard any empty lines
        header_found = False
        headers = []
        
        for line in lines:
                if line.strip().split()[0]=="Step":
                    headers = line.split()  # Column headers
                    header_found = True

                    # Initialize lists for each header in the dictionary
                    data_dict = {header: [] for header in headers}
                    continue

                if header_found:
                    if line.strip().split()[0]=='Loop':
                        break
                    values = line.split()
                    for header, value in zip(headers, values):
                        data_dict[header].append(float(value))

        for header in headers:
            data_dict[header]=np.array(data_dict[header])
            
    return data_dict

def write_log_file_to_csv(log_file, out_path):
    data = read_lammps_log_file(log_file)
    with open(out_path,'w') as doc:
        for i in list(data.keys()):
            doc.write(str(i)+',')
        doc.write('\n')
        for i in range(len(data[list(data.keys())[0]])):
            line=''
            for j in list(data.keys()):
                line+=str(data[j][i])+','
            doc.write(line+'\n')
    doc.close()

def moving_average(input, n):
    L = len(input)
    output = []
    for i in range(L):
        output.append(np.average(input[i:min(i+n,L)]))
    return output

import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(x, y):
    """
    Calculate the autocorrelation function (ACF) of y with respect to x.
    
    Parameters:
        x (array-like): The x data (e.g., time or position).
        y (array-like): The y data to calculate the autocorrelation of.
        
    Returns:
        lags (np.ndarray): Array of lag times.
        acf (np.ndarray): Autocorrelation values for each lag.
    """
    y = np.asarray(y) - np.mean(y)  # Demean y to remove any offset
    n = len(y)
    
    # Calculate full autocorrelation via FFT
    acf_full = np.correlate(y, y, mode='full') / np.var(y) / n
    acf = acf_full[n - 1:]  # Keep only positive lags

 
    dx = x[1] - x[0] 
    lags = np.arange(len(acf)) * dx
    
    return lags, acf