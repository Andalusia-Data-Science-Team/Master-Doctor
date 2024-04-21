from fuzzywuzzy import fuzz

def get_all_db_names(edited_staff,doc_undefined_prep):
    list_of_lists = list(edited_staff['Staff_name']) + list(doc_undefined_prep['Topic'])
    list_of_lists2 = []
    for elem in list_of_lists:
        list_of_lists2 = list_of_lists2 + elem.split(" ")

    list_of_lists2 = list(set(list_of_lists2))
    list_of_lists2 = [elem for elem in list_of_lists2 if len(elem) > 2]
    return list_of_lists2

class NameUnifier:
    def __init__(self, names,thresh=90):
        self.unified_names = self._unify_names(names,thresh)

    def _unify_names(self, name_list, threshold=90):
        unified_names = {}

        for name in name_list:
            matched = False
            for unified_name, score in unified_names.items():
                similarity_score = fuzz.ratio(name, unified_name)
                if similarity_score > threshold:
                    unified_names[unified_name].append(name)
                    matched = True
                    break

            if not matched:
                unified_names[name] = [name]

        return unified_names

    def print_db(self):
        print(self.unified_names)

    def print_substitute(self, name):
        for k, v in self.unified_names.items():
            if name in v:
                print(f'{name} matches the unification: {k}')

    def get_substitute(self, name):
        for k, v in self.unified_names.items():
            if name in v:
                return k

    def get_substitute_full(self, full_name):
        collect = ''
        for name in full_name.split(' '):
            for k, v in self.unified_names.items():
                if (name.lower().strip() not in ["hamed",'ahmed','nehal','wael','mosab']):
                    if name in v:
                        collect = collect + ' ' + k
                else:
                    collect = collect + ' ' + name.strip()
        return collect.strip()