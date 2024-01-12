from table import Table
class Ttable:
  def __init__(self,table,r,c):
    self.table = table
    self.r = r
    self.c = c
  def get_row(self,row_no):return self.table[str(min(row_no,self.r)),:]
  def get_col(self,col_no):return self.table[:,str(min(col_no,self.c))]
  def get_whole(self):return self.table
  def range_row(self,range_row):return self.table[range_row,:]
  def range_col(self,range_col):return self.table[:,range_col]
  def range_row_and_col(self,range_row,range_col):return self.table[range_row,range_col]    
if __name__=="__main__":
  table = Table(
    ["1", "2", "3"], 
    ["a", "b"],      
    [[1, 2],        
     [3, 4],
     [5, 6]]
  tt = Ttable(table,3,2)
  print(tt.get_row(1))