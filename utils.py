def only(array, indexes):
    ref = set()

    for i in indexes:
        ref.add(array[i])

    return all(ref) and not any([x for i,x in enumerate(array) if i not in indexes])
    