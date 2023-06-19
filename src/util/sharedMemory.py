import mmap
import struct
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("mid",type=int,help="missile id")

class WICONEx():
    def __init__(self,bufSize,reqAddr,repAddr,mid,aid):
        self.bufSize=bufSize
        self.mid=mid
        self.aid=aid
        self.reqAddr=reqAddr
        self.repAddr=repAddr
        self.request=0
        self.reply=0
        self.requestlast=0
        self.replylast=0
    #打开数据交换区
    def Open(self):
        self.smname="SM"+str(mid)+str(aid)
        print(self.smname)
        self.shmem=mmap.mmap(0,self.bufSize,self.smname,mmap.ACCESS_WRITE)

    #判断是否存在新的请求，如果不读取请求数据，则一直保持有新的请求状态
    def HasRequest(self):
        self.shmem.seek(0x0)
        res=self.shmem.read(4)
        self.request = struct.unpack("i", res)[0]
        if self.request>=self.requestlast+1:
            return True
        else:
            return False

    #读取请求数据，按长度读取
    def ReceiveRequest(self):
        self.shmem.seek(self.reqAddr)
        res=self.shmem.read(4*50)
        res=struct.unpack("50f",res)
        print(res)
        self.requestlast=self.request
        return res

    #发送反馈数据
    def SendReply(self,reply_data):
        self.shmem.seek(self.repAddr)
        self.shmem.write(reply_data)
        self.reply=self.reply+1
        reply_singal=struct.pack("i",self.reply)
        self.shmem.seek(0x8)
        self.shmem.write(reply_singal)
        