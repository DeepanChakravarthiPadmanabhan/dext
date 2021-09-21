def add_saliency_maps(saliency_list):
    return saliency_list


def merge_saliency(saliency_list, merge_method='add'):
    if merge_method == 'add':
        return add_saliency_maps(saliency_list)
    else:
        raise ValueError("Saliency merge method not implemented %s"
                         % merge_method)
