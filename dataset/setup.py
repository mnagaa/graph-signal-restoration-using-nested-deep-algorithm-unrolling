import itertools

from utils.logger import logger

def setup_community_graph():
    import dataset.builder.denoising_community as dc
    # community
    propsA = dc.CommunityProps(title='dataA', sigma=0.5)
    dc.community_graph_dataset(propsA)
    propsB = dc.CommunityProps(title='dataB', sigma=1.0)
    dc.community_graph_dataset(propsB)
    logger.success("Successfully built the community graph dataset.")

def setup_sensor_graph():
    import dataset.builder.denoising_sensor as ds
    # sensor
    propsA = ds.SensorProps(title='dataA', sigma=0.5)
    ds.sensor_graph_dataset(propsA)
    propsB = ds.SensorProps(title='dataB', sigma=1.0)
    ds.sensor_graph_dataset(propsB)
    logger.success("Successfully built the sensor graph dataset.")

def setup_us_temp():
    # U.S. temperature
    import dataset.builder.denoising_us_temp as dus
    dus.temp2017()

def setup_pointcloud():
    import dataset.builder.denoising_pointcloud as dp
    # pointcloud
    npoints_set = {200, 500, 1_000, 2_000, 5_000, 10_000}
    sigma_set = {10, 20, 30, 40}
    for (_npoints, _sigma) in  itertools.product(npoints_set, sigma_set):
        logger.info(f"Loop start: {_npoints=}, {_sigma=}")
        props = dp.PointCloudProps(npoints=_npoints, sigma=_sigma)
        dp.generate(props)

def setup_sensor_corrupt():
    import dataset.builder.denoising_sensor_corrupt as dsc
    # generate graph corrupt data
    for s, d in [(0.5, 'dataA'), (1.0, 'dataB')]:
        props = dsc.PWS_Props(sigma=s, data_name=d)
        dsc.start_generating(props)
        props = dsc.PWC_Props(sigma=s, data_name=d)
        dsc.start_generating(props)
        props = dsc.GS_Props(sigma=s, data_name=d)
        dsc.start_generating(props)


if __name__ == '__main__':
    setup_community_graph()
    setup_sensor_graph()
    setup_us_temp()
    setup_sensor_corrupt()
    setup_pointcloud()








