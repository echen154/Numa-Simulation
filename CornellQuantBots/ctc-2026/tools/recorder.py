"""Read-only recorder: WS (news, quotes, trades) + REST full order books every 2 s. Places no orders."""
import asyncio, json, time, sys, aiohttp, websockets
B='https://cornelltradingcompetition.org'; import os; K=open(os.path.expanduser(os.environ.get('CTC_KEYS_FILE','~/.ctc/keys.txt'))).readlines()[-1].strip()
SY=['PORT','ISLE','AQUA','GRID','FERRY','RECON','MAT','GEN','HAVEN','REIN','ARCH','REBLD']
H={'User-Agent':'ctc-recorder/1.0','X-API-Key':K}
async def ws(out):
    while True:
        try:
            async with websockets.connect(B.replace('https','wss')+'/ws/v1/market-data',additional_headers=H,max_size=2**22) as w:
                await w.send(json.dumps({"action":"subscribe","symbols":SY,"channels":["news","quotes","trades"]}))
                async for f in w:
                    out.write(json.dumps({'rt':time.time(),'src':'ws','d':json.loads(f)})+'\n')
        except Exception as e:
            out.write(json.dumps({'rt':time.time(),'src':'err','e':repr(e)})+'\n'); await asyncio.sleep(2)
async def books(out):
    async with aiohttp.ClientSession(headers=H) as s:
        while True:
            t0=time.time()
            try:
                async def one(sym):
                    async with s.get(f'{B}/api/v1/orderbook/{sym}',timeout=aiohttp.ClientTimeout(total=5)) as r: return await r.json()
                res=await asyncio.gather(*[one(x) for x in SY],return_exceptions=True)
                async with s.get(f'{B}/api/v1/competitions/time',timeout=aiohttp.ClientTimeout(total=5)) as r: clk=await r.json()
                async with s.get(f'{B}/api/v1/positions',timeout=aiohttp.ClientTimeout(total=5)) as r: pos=await r.json()
                out.write(json.dumps({'rt':t0,'src':'books','clock':clk,'pos':pos,'books':[x if isinstance(x,dict) else repr(x) for x in res]})+'\n')
            except Exception as e:
                out.write(json.dumps({'rt':time.time(),'src':'err','e':repr(e)})+'\n')
            await asyncio.sleep(max(0,2-(time.time()-t0)))
async def main():
    with open(sys.argv[1],'a',buffering=1) as out:
        await asyncio.gather(ws(out),books(out))
asyncio.run(main())
