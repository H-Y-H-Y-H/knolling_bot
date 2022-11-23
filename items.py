import numpy as np
from stl import mesh

class sort():
    
    def __init__(self, num_2x2=0, num_2x3=0, num_2x4=0, num_pencil=0):

        self.correct = []
        self.cube_2x2 = np.array([[0.015, 0.015, 0.012]])
        self.cube_2x3 = np.array([[0.023, 0.015, 0.012]])
        self.cube_2x4 = np.array([[0.033, 0.015, 0.012]])
        self.pencil = np.array([[0.01518, 0.09144, 0.01524]])
        
        if num_2x2 != 0:
            self.correct.append(list(self.cube_2x2))
        if num_2x3 != 0:
            self.correct.append(list(self.cube_2x3))
        if num_2x4 != 0:
            self.correct.append(list(self.cube_2x4))
        if num_pencil != 0:
            self.correct.append(list(self.pencil))
        # self.correct = np.concatenate((self.cube_2x2, self.cube_2x3, self.cube_2x4, self.pencil), axis=0)

        self.error_rate = 0.0001

        self.num_2x2 = num_2x2
        self.num_2x3 = num_2x3
        self.num_2x4 = num_2x4
        self.num_pencil = num_pencil

    def get_data(self, number_kinds):
        
        #! don't motify
        names = globals()
        xyz_list = []
        pos_list = []
        ori_list = []

        for i in range(self.num_2x2):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/cube2x2/cube2x2_%d.STL' % i)
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        for i in range(self.num_2x3):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/cube2x3/cube2x3_%d.STL' % i)
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        for i in range(self.num_2x4):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/cube2x4/cube2x4_%d.STL' % i)
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        for i in range(self.num_pencil):
            names[f'cube_{i}_dimension'] = mesh.Mesh.from_file('urdf/pencil/crayon.STL')
            xyz_list.append(names['cube_%d_dimension' % i].max_ - names['cube_%d_dimension' % i].min_)
        xyz_list = np.asarray(xyz_list, dtype=np.float32)

        return self.judge(xyz_list, pos_list, ori_list, number_kinds)
    
    def judge(self, item_xyz, item_pos, item_ori, number_kinds):

        #! initiate the number of items
        all_index = []
        new_item_xyz = []

        items_names = globals()
        for i in range(number_kinds):
            items_names[f'item_{i}'] = []
            items_names[f'xyz_{i}'] = []
            for j in range(item_xyz.shape[0]):

                if abs(np.sum(item_xyz[j, :] - self.correct[i])) < np.sum(self.correct[i]) * self.error_rate:
                    items_names[f'item_{i}'].append(j)
                    new_item_xyz.append(list(item_xyz[j]))
                    # items_names[f'xyz_{i}'].append(list(item_xyz[j]))
            #! the order of all_index is based on 'self.correct'
            all_index.append(items_names[f'item_{i}'])

        new_item_xyz = np.asarray(new_item_xyz).reshape(-1, 3)
        # for i in range(len(all_index)):

        #     for j in range(len(all_index[i])):

        #         new_item_xyz.append(list(item_xyz[all_index[i][j]]))
        # new_item_xyz = np.asarray(new_item_xyz)
        # print('this is xyz try\n', new_item_xyz)
        # print('this is xyz try\n', item_xyz)

        return new_item_xyz, item_pos, item_ori, all_index

if __name__ == '__main__':

    number_kinds = 3
    SORT = sort(num_2x2=2, num_2x3=3, num_2x4=4, num_pencil=0)
    xyz_list, _, _, all_index = SORT.get_data(number_kinds)
    print(xyz_list)
    print(all_index)
