import pandas as pd
import os

def get_follower_overlap(user_profile_path):
    try:
        with open(user_profile_path, "r", encoding="ISO-8859-1") as file:
            file_contents = file.read()

        # Split the file contents by blank lines
        data_blocks = file_contents.strip().split("\n\n")

        # Parse bi_followers_count and followers_count into a list of tuples
        uids = []
        bi_followers_counts = []
        followers_counts = []
        ratio = []
        errors = []
        for block in data_blocks[1:]:
            try:
                lines = block.split("\n")

                # special case when last feature is empty (empty description), 
                # so it becomes part of the next row during parsing
                # so we need to discard it from current row
                if lines[0] == '':
                    lines.pop(0)

                user = lines[0]
                bi_fol = int(lines[1])
                fol = int(lines[4])

                r = 1.0 * bi_fol / fol if fol > 0 else 0
                uids.append(user)
                bi_followers_counts.append(bi_fol)
                followers_counts.append(fol)
                ratio.append(r)
                assert r >= 0 and r <= 1
                print(f"Procssed user {user} with ratio {r}")
            except:
                errors.append(user)
        
        df = pd.DataFrame(
                {"UID": uids,
                "bi_followers_counts": bi_followers_counts,
                "followers_counts": followers_counts,
                "follower_overlap_ratio": ratio
                })
        
        fName = user_profile_path.split('/')[-1] if '/' in user_profile_path \
                                                else user_profile_path.split('\\')[-1] 
        fName_wo_ext = fName.split('.')[0]
        path = os.getcwd() + "_" + fName_wo_ext + ".csv"
        df.to_csv(path, index=False)
        print(f"Storing results into csv, path: {path}")
        print(f"Some users could not be processed: {errors}")
    
    except FileNotFoundError:
        print(f"File not found: {user_profile_path}")
        return []
    except Exception as e:
        print(f"Something went wrong: {e}")

get_follower_overlap("E:\\w210\\Data\\Data\\Weibo\\Init_Data\\userProfile\\user_profile1.txt")
# get_follower_overlap("E:\\w210\\Data\\Data\\Weibo\\Init_Data\\userProfile\\user_profile2.txt")